#!/usr/bin/env bash


stage=3
nj=6
data=MTG1_dev_mc_plaza0
affix=MTG_gss
train_set_speaker_embedding_list=./data/train_ivectors_spk.txt
# feature_list=wav_fbank.list
feature_list=wav_fbank2.list

model_path=/train33/sppro/permanent/stniu/MAMSE_sep/exp/S2S/Batchsize8_8speakers_Segment800s_Mixup0.0_NOTSOFAR_simulated_all_data_512_all0Dropout_6layers_weight_input_separation_ov
# model_config=configs_4Speakers_wavlm_mamse
model_config=configs3_4Speakers_ivector_ivector128_xvectors128_S2S_MA_MSE_DIM_chime8
diarized_rttm=/train33/sppro/permanent/stniu/MAMSE_sep/data/MTG2_dev_mc/0409_s4.rttm
oracle_vad=
dihard_path=/train20/intern/permanent/mkhe/tools/kaldi-master/egs/tsvad
output_dir_sep=/train33/sppro/permanent/stniu/NOTSOFAR1-Challenge-main-mask/artifacts/240208.2_train_css_baseline_tsvad_simu
#num_channels=15
#oracle_vad=data/chime7_dev_array/tdnn_vad.lab
#oracle_vad=../data/CHiME7/chime6/oracle_vad/dev.lab

# max_utt_durance=200
max_utt_durance=186
max_speaker=8
batch_size=4
ivector_dir=exp/nnet3_recipe_ivector
do_vad=false
gpu=5
input_gpu_id=1

# exit
model_type=model_S2S_weight_input_DIM_css_conformer_noise_maskloss_dia_jointrain_sc
hop_len=93
th_s=10
th_e=60
frame_per_second=62.5

min_dur=0.2
segment_padding=0.10
max_dur=0.80

. path.sh
. utils/parse_options.sh

gpu=$((input_gpu_id - 1))
echo $input_gpu_id
echo $gpu
echo $model_path
echo $stage
echo $max_utt_durance
echo $hop_len

if [ $stage -le 1 ]; then
  # for l in `cat data/$data/utt2spk`;do 
      # s=`echo $l | cut -d "_" -f 1`
      # grep $s $diarized_rttm | awk -v var=$l '{$2=var;print $0}'
    # done > data/$data/diarized_fusion.rttm
  local/extract_feature.sh --stage 3 --nj $nj \
      --sample_rate _16k --ivector_dir $ivector_dir \
      --max_speaker $max_speaker --affix _$affix \
      --rttm $diarized_rttm --data $data
fi

if [ $stage -le 2 ]; then
    
    CUDA_VISIBLE_DEVICES=$gpu python local_gb/decode_dcfds_taslp_os.py \
        --feature_list data/${data}/${feature_list}${gpu} \
        --embedding_list ./data/ivectors_MTG_eval_sc_sq_0711/ivectors_spk.txt \
        --train_set_speaker_embedding_list ${train_set_speaker_embedding_list} \
        --model_path ${model_path} \
        --output_dir ${model_path}_${data}_${affix} \
        --output_dir_sep $output_dir_sep \
        --max_speaker $max_speaker \
        --init_rttm ${diarized_rttm} \
        --model_config ${model_config} \
        --max_utt_durance ${max_utt_durance} \
        --hop_len ${hop_len} \
        --batch_size $batch_size \
        --gpu $gpu \
        --remove_overlap \
        --model_type $model_type \
        --frame_per_second $frame_per_second
fi

exit 