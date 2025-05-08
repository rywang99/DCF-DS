. ./path.sh
script_lmdb=script_decode
model_type=model_S2S_weight_input_DIM_css_conformer_noise_maskloss_dia_jointrain_800_stage2_sc

model_path=./MIMO_SE.model

split_num=4
max_utt_durance=800
hop_len=100


rm -rf ./$script_lmdb/*
for i in `seq 1 ${split_num}`; do echo  sh local_gb/decode_mimose_taslp_os.sh --model_path $model_path --stage 2 --data MTG1_eval_sc --feature_list wav_fbank.list \
--input_gpu_id $i --max_utt_durance $max_utt_durance --hop_len $hop_len --model_type $model_type --batch_size 4 \
--output_dir_sep /train33/sppro/permanent/stniu/NOTSOFAR1-Challenge-main-mask-dia/artifacts/redev_outputs_MTG1_plaza0_v2/css_real_len${max_utt_durance}_hop${hop_len}_800_eval_mask_sc_0715_stage2_os_test \
> $script_lmdb/do_getlmdb.sh$i; done
chmod -R 777 ./$script_lmdb

run.pl JOB=1:${split_num} ${script_lmdb}/log/do_getlmdb.shJOB.log ${script_lmdb}/do_getlmdb.shJOB
