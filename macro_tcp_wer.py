import glob
import json

srcdir = "/train33/sppro/permanent/rywang9/NOTSOFAR1-Challenge-DEBUG/artifacts/redev_outputs/MTG2_dev_mc-0409_odcssrec_large-v3"
srcdir = '/train33/sppro/permanent/stniu/NOTSOFAR1-Challenge-main-mask/artifacts2/MTG2_dev_mc-0318_v3rec_before_sd_large-v3'
srcdir = 'artifacts/css_real_len200_hop100_model8_dia_maskth0.55_diath0.55'
json_list = glob.glob(f"{srcdir}/wer/multichannel/*/tcp_wer_hyp_tcpwer.json")

total_tcp_wer = 0
total_errors = 0
total_length = 0
total_insertions = 0
total_deletions = 0
total_substitutions = 0

for file in json_list:
    with open(file) as f:
        data = json.load(f)
    total_tcp_wer = total_tcp_wer + data["error_rate"]
    total_errors += data["errors"]
    total_length += data["length"]
    total_insertions += data["insertions"]
    total_deletions += data["deletions"]
    total_substitutions += data["substitutions"]

macro_tcp_wer = total_tcp_wer / len(json_list)
print("macro_tcp_wer: ", macro_tcp_wer)
print("total_length: ", total_length)
print("total_errors: ", total_errors)
print("total_insertions: ", total_insertions)
print("total_deletions: ", total_deletions)
print("total_substitutions: ", total_substitutions)
