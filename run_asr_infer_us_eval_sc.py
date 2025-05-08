import argparse
from pathlib import Path
from pprint import pprint
from typing import Literal
import sys

from inference_pipeline.inference_sc import (
    InferenceCfg,
    inference_pipeline_us,
    FetchFromCacheCfg,
)
from utils.azure_storage import download_meeting_subset, download_models
from utils.conf import load_yaml_to_dataclass, update_dataclass

ConfigName = Literal["full_dev_set_mc", "full_dev_set_sc", "dev_set_1_mc_debug"]
import warnings

warnings.filterwarnings("ignore")


def get_project_root() -> Path:
    """Returns project root folder"""
    return Path(__file__).parent


def load_config(config_name: ConfigName) -> InferenceCfg:
    """Returns the config file path and session query for the given config name"""
    project_root = get_project_root()

    updates = {}
    if config_name == "full_dev_set_mc":
        # all multi-channel (MC) dev-set sessions
        conf_file = project_root / "configs/inference/inference_v1.yaml"
        session_query = 'is_mc == False' # filter only MC

    elif config_name == "full_dev_set_sc":
        # all single-channel (SC) dev-set sessions
        conf_file = project_root / "configs/inference/inference_v1.yaml"
        session_query = "is_mc == False"  # filter only SC

    elif config_name == "dev_set_1_mc_debug":
        # for quick debug: 'tiny' Whisper, one MC (multi-channel) session
        conf_file = project_root / "configs/inference/debug_inference.yaml"
        session_query = (
            'device_name == "plaza_0" and is_mc == True and meeting_id == "MTG_30860"'
        )

    else:
        raise ValueError(f"unknown config name: {config_name}")

    cfg: InferenceCfg = load_yaml_to_dataclass(str(conf_file), InferenceCfg)
    cfg = update_dataclass(cfg, updates)

    if session_query is not None:
        assert cfg.session_query is None, "overriding session_query from yaml"
        cfg.session_query = session_query

    return cfg


def main(
    config_name: ConfigName = "dev_set_1_mc_debug",
    exp_name: str = "",
    gss_enhanced_dir: str = "",
    asr_model: str = "",
):
    project_root = get_project_root()
    cfg: InferenceCfg = load_config(config_name)

    cfg.asr.model_name = asr_model

    # TODO: ROOT DIR
    # dev_meetings_dir = '/yrfs5/sre/sqqian/data/chime8_dev/240415.2_dev_with_GT/MTG'
    dev_meetings_dir = '/train33/sppro/permanent/stniu/data/chime8/chime8_eval/240629.1_eval_small_with_GT/MTG'
    models_dir = '/train33/sppro/permanent/stniu/NOTSOFAR1-Challenge-main/artifacts/css_models'
    outputs_dir = project_root / "artifacts" / "redev_outputs_MTG1_plaza0_v2"

    cache_cfg = FetchFromCacheCfg()  # no cache, use this at your own risk.

    exp_dir = exp_name
    outputs_dir = outputs_dir / exp_dir

    pprint(f"{config_name=}")
    pprint(cfg)

    # run infesrence pipeline
    inference_pipeline_us(
        meetings_dir=str(dev_meetings_dir),
        models_dir=str(models_dir),
        out_dir=str(outputs_dir),
        cfg=cfg,
        cache=cache_cfg,
        gss_enhanced_dir=gss_enhanced_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference pipeline")
    parser.add_argument(
        "--config-name",
        type=str,
        default="full_dev_set_mc",
        help="Config scenario for the inference, default: dev_set_1_mc_debug",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="css",
        help="Output directory path, default: ./artifacts/outputs",
    )
    parser.add_argument(
        "--gss_enhanced_dir",
        type=str,
    )
    parser.add_argument(
        "--asr_model",
        type=str,
    )
    args = parser.parse_args()

    main(args.config_name, args.exp_name, args.gss_enhanced_dir, args.asr_model)
