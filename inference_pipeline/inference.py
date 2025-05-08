from dataclasses import field, dataclass
from typing import Optional
import os
import torch
import multiprocessing
import tqdm
import pandas as pd
import glob
import whisper
from asr.asr import (
    asr_inference,
    WhisperAsrCfg,
    gss_asr_inference,
    gss_asr_inference_preload_model,
)
from css.css import css_inference, CssCfg
from diarization.diarization import diarization_inference
from diarization.diarization_common import DiarizationCfg
from inference_pipeline.load_meeting_data import load_data
from utils.logging_def import get_logger
from utils.scoring import ScoringCfg, calc_wer, write_transcript_to_stm

_LOG = get_logger("inference")


def split_dataframe_randomly(df: pd.DataFrame, num_subdfs):
    # 将DataFrame中的行进行随机重排
    df_shuffled = df.sample(frac=1).reset_index(drop=True)

    # 计算每个子DataFrame的大小
    subdf_size = len(df) // num_subdfs
    remaining_rows = len(df) % num_subdfs

    # 划分成子DataFrame
    subdfs = []
    start_index = 0
    for i in range(num_subdfs):
        if i < remaining_rows:
            end_index = start_index + subdf_size + 1
        else:
            end_index = start_index + subdf_size
        subdf = df_shuffled.iloc[start_index:end_index]
        subdfs.append(subdf)
        start_index = end_index

    return subdfs


@dataclass
class InferenceCfg:
    css: CssCfg = field(default_factory=CssCfg)
    asr: WhisperAsrCfg = field(default_factory=WhisperAsrCfg)
    diarization: DiarizationCfg = field(default_factory=DiarizationCfg)
    scoring: ScoringCfg = field(default_factory=ScoringCfg)
    # Optional: Query to filter all_session_df. Useful for debugging. Must be None during full evaluation.
    session_query: Optional[str] = None


@dataclass
class FetchFromCacheCfg:
    css: bool = True
    asr: bool = True
    diarization: bool = True


def inference_pipeline(
    meetings_dir: str,
    models_dir: str,
    out_dir: str,
    cfg: InferenceCfg,
    cache: FetchFromCacheCfg,
):
    """
    Run the inference pipeline on sessions loaded from meetings_dir.

    Args:
        meetings_dir: directory with meeting data.
            example: project_root/artifacts/meeting_data/dev_set/240121_dev/MTG/
        models_dir: directory with CSS models.
            example: project_root/artifacts/css_models/
        out_dir: modules will write their outputs here.
        cfg: config per module.
        cache: basic cache mechanism to re-use results per module. Off by default.
            Note: use at your own risk. If you modify code or config, make sure to delete the cache
            or set to False.
    """
    # Load all meetings from the meetings dir
    _LOG.info(f"loading meetings from: {meetings_dir}")
    all_session_df, all_gt_utt_df, all_gt_metadata_df = load_data(
        meetings_dir, cfg.session_query
    )

    # Process each session independently. (Cross-session information is not permitted)
    wer_series_list = []
    for session_name, session in tqdm.tqdm(
        all_session_df.iterrows(), desc="processing sessions"
    ):
        _LOG.info(f"Processing session: {session.session_id}")

        # Front-end: split session into enhanced streams without overlap speech
        session: pd.Series = css_inference(
            out_dir, models_dir, session, cfg.css, cache.css
        )

        # Run ASR on each stream and return transcribed segments
        segments_df: pd.DataFrame = asr_inference(out_dir, session, cfg.asr, cache.asr)

        # Return speaker attributed segments (re-segmentation can occur)
        attributed_segments_df: pd.DataFrame = diarization_inference(
            out_dir, segments_df, cfg.diarization, cache.diarization
        )

        # Write hypothesis transcription to: outdir / wer / {multi|single}channel /.../ *.stm
        # To submit your system for evaluation, send us the contents of: outdir / wer / {multi|single}channel
        tcp_wer_hyp_stm, tcorc_wer_hyp_stm = write_hyp_transcripts(
            out_dir,
            session.session_id,
            attributed_segments_df,
            attributed_segments_df,
            cfg.asr.text_normalizer(),
        )

        # Calculate WER if GT is available
        if all_gt_utt_df is not None:
            # Rules: WER metric, arguments (collar), and text normalizer must remain unchanged
            session_wer: pd.Series = calc_wer(
                out_dir,
                tcp_wer_hyp_stm,
                tcorc_wer_hyp_stm,
                session.session_id,
                get_session_gt(session, all_gt_utt_df),
                cfg.asr.text_normalizer(),
                collar=5,
                save_visualizations=cfg.scoring.save_visualizations,
            )
            wer_series_list.append(session_wer)

    if wer_series_list:
        # if GT is available, aggregate WER.
        all_session_wer_df = pd.DataFrame(wer_series_list)
        _LOG.info(f"Results:\n{all_session_wer_df}")
        _LOG.info(f'mean tcp_wer = {all_session_wer_df["tcp_wer"].mean()}')
        _LOG.info(f'mean tcorc_wer = {all_session_wer_df["tcorc_wer"].mean()}')

        # write session level results into a file
        exp_id = "_".join(
            [
                os.path.basename(cfg.css.checkpoint_sc),
                cfg.asr.model_name,
                cfg.diarization.method,
            ]
        )
        os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)
        result_file = os.path.join(out_dir, "results", exp_id + ".tsv")
        _LOG.info(f"Results can be found in: {result_file}")
        all_session_wer_df.to_csv(result_file, sep="\t")
        # TODO confidence intervals, WER per meta-data


def subdf_inference(
    session_df,
    all_gt_utt_df,
    out_dir: str,
    cfg: InferenceCfg,
    cache: FetchFromCacheCfg,
    gpu_id,
    gss_enhanced_dir: str,
):
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda")
    _LOG.info(f"Loading Whisper model: {cfg.asr.model_name}")
    model = whisper.load_model(
        cfg.asr.model_name, device, download_root="whisper_model"
    )
    session_wer_list = []
    for session_name, session in tqdm.tqdm(
        session_df.iterrows(), desc="processing sessions"
    ):
        _LOG.info(f"Processing session: {session.session_id}")
        _LOG.info(f"{gss_enhanced_dir}/{session.meeting_id}-mc_{session.device_name}")
        session["sep_wav_file_names"] = glob.glob(
            f"{gss_enhanced_dir}/{session.meeting_id}_{session.device_name}/*.wav"
        )
        attributed_segments_df: pd.DataFrame = gss_asr_inference_preload_model(
            out_dir, session, cfg.asr, cache.asr, model
        )
        tcp_wer_hyp_stm, tcorc_wer_hyp_stm = write_hyp_transcripts(
            out_dir,
            session.session_id,
            attributed_segments_df,
            attributed_segments_df,
            cfg.asr.text_normalizer(),
        )
        if all_gt_utt_df is not None:
            # Rules: WER metric, arguments (collar), and text normalizer must remain unchanged
            session_wer: pd.Series = calc_wer(
                out_dir,
                tcp_wer_hyp_stm,
                tcorc_wer_hyp_stm,
                session.session_id,
                get_session_gt(session, all_gt_utt_df),
                cfg.asr.text_normalizer(),
                collar=5,
                save_visualizations=cfg.scoring.save_visualizations,
            )
            session_wer_list.append(session_wer)
    return session_wer_list


def inference_pipeline_us(
    meetings_dir: str,
    models_dir: str,
    out_dir: str,
    cfg: InferenceCfg,
    cache: FetchFromCacheCfg,
    gss_enhanced_dir: str,
):
    """
    Run the inference pipeline on sessions loaded from meetings_dir.

    Args:
        meetings_dir: directory with meeting data.
            example: project_root/artifacts/meeting_data/dev_set/240121_dev/MTG/
        models_dir: directory with CSS models.
            example: project_root/artifacts/css_models/
        out_dir: modules will write their outputs here.
        cfg: config per module.
        cache: basic cache mechanism to re-use results per module. Off by default.
            Note: use at your own risk. If you modify code or config, make sure to delete the cache
            or set to False.
    """
    # Load all meetings from the meetings dir
    _LOG.info(f"loading meetings from: {meetings_dir}")
    all_session_df, all_gt_utt_df, all_gt_metadata_df = load_data(
        meetings_dir, cfg.session_query
    )

    num_gpus = torch.cuda.device_count()

    pool = multiprocessing.Pool(processes=num_gpus)
    subs_session_df_list = split_dataframe_randomly(all_session_df, num_gpus)
    results = []
    # 在每个GPU上启动一个进程，处理对应的子列表，并获取返回值
    for gpu_id, subdf in enumerate(subs_session_df_list):
        # subdf_inference(subdf, all_gt_utt_df, out_dir, cfg, cache, gpu_id, gss_enhanced_dir)
        result = pool.apply_async(
            subdf_inference,
            args=(subdf, all_gt_utt_df, out_dir, cfg, cache, gpu_id, gss_enhanced_dir),
        )
        results.append(result)
    pool.close()
    pool.join()
    wer_series_list = []
    for result in results:
        wer_series_list.append(result.get())

    # # Process each session independently. (Cross-session information is not permitted)
    # wer_series_list = []
    # for session_name, session in tqdm.tqdm(
    #     all_session_df.iterrows(), desc="processing sessions"
    # ):
    #     _LOG.info(f"Processing session: {session.session_id}")
    #     session["sep_wav_file_names"] = glob.glob(
    #         f"{gss_enhanced_dir}/{session.meeting_id}-mc_{session.device_name}/*.wav"
    #     )
    #     # Front-end: split session into enhanced streams without overlap speech
    #     # session: pd.Series = css_inference(
    #     #     out_dir, models_dir, session, cfg.css, cache.css
    #     # )

    #     # Run ASR on each stream and return transcribed segments
    #     attributed_segments_df: pd.DataFrame = gss_asr_inference(
    #         out_dir, session, cfg.asr, cache.asr
    #     )

    #     # Return speaker attributed segments (re-segmentation can occur)
    #     # attributed_segments_df: pd.DataFrame = diarization_inference(
    #     #     out_dir, segments_df, cfg.diarization, cache.diarization
    #     # )

    #     # Write hypothesis transcription to: outdir / wer / {multi|single}channel /.../ *.stm
    #     # To submit your system for evaluation, send us the contents of: outdir / wer / {multi|single}channel
    #     tcp_wer_hyp_stm, tcorc_wer_hyp_stm = write_hyp_transcripts(
    #         out_dir,
    #         session.session_id,
    #         attributed_segments_df,
    #         attributed_segments_df,
    #         cfg.asr.text_normalizer(),
    #     )

    #     # Calculate WER if GT is available
    #     if all_gt_utt_df is not None:
    #         # Rules: WER metric, arguments (collar), and text normalizer must remain unchanged
    #         session_wer: pd.Series = calc_wer(
    #             out_dir,
    #             tcp_wer_hyp_stm,
    #             tcorc_wer_hyp_stm,
    #             session.session_id,
    #             get_session_gt(session, all_gt_utt_df),
    #             cfg.asr.text_normalizer(),
    #             collar=5,
    #             save_visualizations=cfg.scoring.save_visualizations,
    #         )
    #         wer_series_list.append(session_wer)

    wer_series_list = sum(wer_series_list, [])
    if wer_series_list:
        # if GT is available, aggregate WER.
        all_session_wer_df = pd.DataFrame(wer_series_list)
        
        _LOG.info(f"Results:\n{all_session_wer_df}")
        # _LOG.info(f'mean tcp_wer = {all_session_wer_df["tcp_wer"].mean()}')
        #_LOG.info(f'mean tcorc_wer = {all_session_wer_df["tcorc_wer"].mean()}')

        all_session_wer_df = all_session_wer_df.sort_values(by='session_id')
        _LOG.info(f'Results:\n{all_session_wer_df}')
        _LOG.info(f'mean tcp_wer = {all_session_wer_df["tcp_wer"].mean()}')
        # _LOG.info(f'mean tcorc_wer = {all_session_wer_df["tcorc_wer"].mean()}')
        _LOG.info(f'sum tcp_length = {all_session_wer_df["tcp_length"].sum()}')
        _LOG.info(f'sum tcp_errors = {all_session_wer_df["tcp_errors"].sum()}')
        _LOG.info(f'sum tcp_insertions = {all_session_wer_df["tcp_insertions"].sum()}')
        _LOG.info(f'sum tcp_deletions = {all_session_wer_df["tcp_deletions"].sum()}')
        _LOG.info(f'sum tcp_substitutions = {all_session_wer_df["tcp_substitutions"].sum()}')
        _LOG.info(f'sum tcp_falarm_speaker = {all_session_wer_df["tcp_falarm_speaker"].sum()}')
        _LOG.info(f'sum tcp_missed_speaker = {all_session_wer_df["tcp_missed_speaker"].sum()}')
        _LOG.info(f'sum tcp_scored_speaker = {all_session_wer_df["tcp_scored_speaker"].sum()}')

        
        all_session_wer_df = all_session_wer_df.drop(['tcp_wer_hyp_stm',
                    'tcorc_wer_hyp_stm',
                    'ref_stm']     , axis=1)
        
        # write session level results into a file
        exp_id = "_".join(
            [
                os.path.basename(cfg.css.checkpoint_sc),
                cfg.asr.model_name,
                cfg.diarization.method,
            ]
        )
        os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)
        result_file = os.path.join(out_dir, "results", exp_id + ".tsv")
        _LOG.info(f"Results can be found in: {result_file}")
        all_session_wer_df.to_csv(result_file, sep="\t")
        
        
        result_avg = os.path.join(out_dir, "results", exp_id+".txt")
        with open(result_avg, 'w') as fw:
            fw.write(f'mean tcp_wer = {all_session_wer_df["tcp_wer"].mean()}\n')
            # fw.write(f'mean tcorc_wer = {all_session_wer_df["tcorc_wer"].mean()}\n')
            fw.write(f'sum tcp_length = {all_session_wer_df["tcp_length"].sum()}\n')
            fw.write(f'sum tcp_errors = {all_session_wer_df["tcp_errors"].sum()}\n')
            fw.write(f'sum tcp_insertions = {all_session_wer_df["tcp_insertions"].sum()}\n')
            fw.write(f'sum tcp_deletions = {all_session_wer_df["tcp_deletions"].sum()}\n')
            fw.write(f'sum tcp_substitutions = {all_session_wer_df["tcp_substitutions"].sum()}\n')
            fw.write(f'sum tcp_falarm_speaker = {all_session_wer_df["tcp_falarm_speaker"].sum()}\n')
            fw.write(f'sum tcp_missed_speaker = {all_session_wer_df["tcp_missed_speaker"].sum()}\n')
            fw.write(f'sum tcp_scored_speaker = {all_session_wer_df["tcp_scored_speaker"].sum()}\n')

        _LOG.info(f"Results can be found in: {result_file}")
        all_session_wer_df.to_csv(result_file, sep="\t")
        # TODO confidence intervals, WER per meta-data


def get_session_gt(session: pd.Series, all_gt_utt_df: pd.DataFrame):
    return all_gt_utt_df[all_gt_utt_df["meeting_id"] == session.meeting_id]


def write_hyp_transcripts(
    out_dir,
    session_id,
    attributed_segments_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    text_normalizer,
):
    _LOG.info(f"Writing hypothesis transcripts for session {session_id}")
    # hyp file for tcpWER, the metric used for ranking.
    # MeetEval requires stream _id, which for tcpWER is the same as speaker_id.
    df = attributed_segments_df.copy()
    df["stream_id"] = df["speaker_id"]
    tcp_wer_hyp_stm = write_transcript_to_stm(
        out_dir, df, text_normalizer, session_id, "tcp_wer_hyp.stm"
    )
    _LOG.info(f"tcpwer STM: {tcp_wer_hyp_stm}")

    # hyp file for tcORC-WER, a supplementary metric for analysis.
    # MeetEval requires stream _id, which for tcORC-WER depends on the system.
    # In NOTSOFAR we define the streams as the outputs of CSS (continuous speech separation).
    # If your system does not have CSS you need to define the streams differently.
    # For example: for end-to-end multi-talker ASR you might use a single stream.
    # Overlap speech should go into different streams,
    # or appear in one stream but respecting the order in reference. See https://github.com/fgnt/meeteval.

    # Take wav_file_name from segments_df, rather than attributed_segments_df, since the latter is a result of
    # diarizations, where the segments are built of words potentially coming from different channels.
    # So, in the general case there is no meaningful "channel" that can be associated with a segment.

    # Use segment_df for tcorc_wer
    # df = segments_df.copy()
    df = attributed_segments_df.copy()
    # Use factorize to map each unique wav_file_name to an index.
    # df["stream_id"], uniques = pd.factorize(df["wav_file_name"], sort=True)
    df["stream_id"] = 0
    # _LOG.debug(f"Found {len(uniques)} streams for tc_orc_wer_hyp.stm")
    tcorc_wer_hyp_stm = write_transcript_to_stm(
        out_dir, df, text_normalizer, session_id, "tc_orc_wer_hyp.stm"
    )
    _LOG.info(f"tcorc_wer STM: {tcorc_wer_hyp_stm}")
    return tcp_wer_hyp_stm, tcorc_wer_hyp_stm
