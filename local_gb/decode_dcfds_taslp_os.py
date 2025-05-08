
#from cProfile import label
import os
import numpy as np
import torch
import tqdm
import argparse

import HTK
import config
import utils
from reader_sc_s2s_libricss_join_training_css import RTTM_to_Speaker_Mask
import importlib
import pdb
import scipy.io.wavfile as sio
from css_conformer.training.losses import mse_loss, PitWrapper, l1_loss
from css_conformer.utils.audio_utils import write_wav
from css_conformer.utils.numpy_utils import dilate, erode
from css.css_with_conformer.utils.mvdr_util import make_mvdr
from css.helpers import load_css_model, load_audio
import torch.nn.functional as Fun
import utils_s2s
from pathlib import Path
import soundfile as sf
import torch.nn.functional as F
from utils_css.plot_utils import plot_stitched_masks, plot_left_right_stitch, plot_separation_methods
import re
# from model_S2S_weight_input_DIM_libricss_join_training_css_conformer_ov import MULTI_MAM_SE_S2S_model as MULTI_MAM_SE_S2S_model_fix_blind
from config import configs3_3Speakers_ivector_ivector128_xvectors128_S2S_MA_MSE_DIM as config_train
from model_S2S_weight_input_DIM_css_conformer_noise_maskloss_dia_jointrain_800_sc_decode2 import MULTI_MAM_SE_S2S_model as MULTI_MAM_SE_S2S_model_fix
from config import configs3_4Speakers_ivector_ivector128_xvectors128_S2S_MA_MSE_DIM_chime8 as config_train2
# from tqdm import tqdm



def plot_mask(label_data, name, plot_flag):
    # ch = 1
    import matplotlib.pyplot as plt
    import soundfile as sf
    
    for i in range(3):
        # pdb.set_trace()
        bb = label_data[i][np.newaxis,:].numpy()
        plot_label = bb.repeat(200, axis = 0)
        # plot_speaker_mask = speaker_mask[i].T
        plt.figure()
        plt.imshow( plot_label, cmap='binary' , interpolation = 'none')
        # if i == 0 and plot_flag:
        plt.colorbar()
        # plt.colorbar()
        plt.savefig(f'./sample/mask_label_{name}_spk{i}.png')
        


def plot_mask_tf(label_data, name, plot_flag):
    # ch = 1
    import matplotlib.pyplot as plt
    import soundfile as sf
    
    for i in range(3):
        # pdb.set_trace()
        plot_label = label_data[i]
        # plot_label = bb.repeat(200, axis = 0)
        # plot_speaker_mask = speaker_mask[i].T
        plt.figure()
        plt.imshow( plot_label, cmap='binary' , interpolation = 'none')
        # if i == 0 and plot_flag:
        plt.colorbar()
        # plt.colorbar()
        plt.savefig(f'./sample/mask_label_{name}_spk{i}.png')
        
        
def plot_mask_tf_long(label_data, name, num_spks):
    # ch = 1
    # pdb.set_trace()
    import matplotlib.pyplot as plt
    import soundfile as sf
    
    for i in range(num_spks):
        # pdb.set_trace()
        plot_label = label_data[:, :, i]
        plot_label = torch.flip(plot_label, [0])
        # plot_label = plot_label.repeat(1, 256)
        # pdb.set_trace()
        # plot_label = bb.repeat(200, axis = 0)
        # plot_speaker_mask = speaker_mask[i].T
        plt.figure()
        plt.imshow( plot_label.cpu(), cmap='jet' , interpolation = 'none')
        # if i == 0 and plot_flag:
        # plt.colorbar()
        # plt.colorbar()
        plt.savefig(f'./sample/{name}_spk{i}.png')

def plot_mask_all(label_data, name, num_speaker):
    # ch = 1
    import matplotlib.pyplot as plt
    import soundfile as sf
    
    for i in range(num_speaker):
        # pdb.set_trace()
        bb = label_data[i][np.newaxis,:]
        plot_label = bb.repeat(2000, axis = 0)
        # plot_speaker_mask = speaker_mask[i].T
        plt.figure()
        plt.imshow( plot_label, cmap='binary' , interpolation = 'none')
        # if i == 0 and plot_flag:
        plt.colorbar()
        # plt.colorbar()
        plt.savefig(f'./sample/mask_label_{name}_spk{i}.png')


def load_ivector(speaker_embedding_txt):
    SCP_IO = open(speaker_embedding_txt)
    speaker_embedding = {}
    raw_lines = [l for l in SCP_IO]
    SCP_IO.close()
    speaker_embedding_list = []
    for i in range(len(raw_lines) // 2):
        speaker = raw_lines[2*i].split()[0]
        session = "-".join(speaker.split("-")[:-1])
        real_speaker = speaker.split("-")[-1]
        if session not in speaker_embedding.keys():
            speaker_embedding[session] = {}
        ivector = torch.from_numpy(np.array(raw_lines[2*i+1].split()[:-1], np.float32))
        speaker_embedding[session][real_speaker] = ivector
        speaker_embedding_list.append(ivector)
    return speaker_embedding, speaker_embedding_list


def load_htk(path):
    nSamples, sampPeriod, sampSize, parmKind, data = HTK.readHtk(path)
    htkdata = np.array(data).reshape(nSamples, int(sampSize / 4))
    #print(nSamples)
    return nSamples, htkdata

def load_single_channel_feature(file_path, window_len=800, hop_len=400):
    nSamples, htkdata = load_htk(file_path)
    # htkdata: T * F
    pdb.set_trace()
    cur_frame, feature, intervals, total_frame = 0, [], [], nSamples
    while(cur_frame < total_frame):
        if cur_frame + window_len <= total_frame:
            feature.append(htkdata[cur_frame:cur_frame+window_len, : ])
            intervals.append((cur_frame, cur_frame+window_len))
            cur_frame += hop_len
        else:
            start = max(0, total_frame-window_len)
            feature.append(htkdata[start:total_frame, : ])
            intervals.append((start, total_frame))
            cur_frame += window_len
    return feature, intervals, total_frame


def load_single_channel_feature_fbank_mcwav_odd(file_path, window_len=1024, hop_len=256):
    # pdb.set_trace()
    wav_path, fbank_path = file_path.split()
    
    # wav_path = wav_path.replace('ch0', f'ch{str(ch)}')
    # pdb.set_trace()
    wav_data_list = []
    for ch in range(7):
        cur_wav_path = wav_path.replace('ch0', f'ch{str(ch)}')
        sr, cur_wav = sio.read(cur_wav_path)
        wav_data_list.append(cur_wav)
    # pdb.set_trace()
    wav = np.vstack(wav_data_list)
    
    
    nSamples1, htkdata = load_htk(fbank_path)
    sample_per_frame = 256
    # wav = HTK.read_wav_start_end(wav_path, 0, nSamples1 * 128 + 80)
    # nSamples2 = (len(wav) - 80) // 320
    # htkdata: T * F
    # nSamples = min(nSamples1 // 2, nSamples2)
    nSamples = int((wav.shape[-1] - 80) / sample_per_frame)
    nSamples = min(nSamples1, nSamples)
    # pdb.set_trace()
    # wav = wav / 32768.
    cur_frame, wav_fea, fbank_fea, intervals, total_frame = 0, [], [], [], nSamples
    while(cur_frame < total_frame):
        if cur_frame + window_len <= total_frame:
            # print(cur_frame*sample_per_frame, (cur_frame+window_len)*sample_per_frame+80)
            wav_fea.append(wav[:, cur_frame*sample_per_frame:(cur_frame+window_len)*sample_per_frame+80])
            fbank_fea.append(htkdata[cur_frame:(cur_frame+window_len), :])
            intervals.append((cur_frame, cur_frame+window_len))
            cur_frame += hop_len
        else:
            start = max(0, total_frame-window_len)
            wav_fea.append(wav[:, start*sample_per_frame:total_frame*sample_per_frame+80])
            fbank_fea.append(htkdata[start:total_frame, : ])
            intervals.append((start, total_frame))
            cur_frame += window_len
    # pdb.set_trace()
    return wav_fea, fbank_fea, intervals, total_frame



def load_single_channel_feature_fbank_mcwav(file_path, window_len=1024, hop_len=256, wav_fea_all_stft=0):
    # pdb.set_trace()
    wav_path, fbank_path = file_path.split()
    
    # wav_path = wav_path.replace('ch0', f'ch{str(ch)}')
    # pdb.set_trace()
    wav_data_list = []
    for ch in range(7):
        cur_wav_path = wav_path.replace('ch0', f'ch{str(ch)}')
        sr, cur_wav = sio.read(cur_wav_path)
        wav_data_list.append(cur_wav)
    # pdb.set_trace()
    wav = np.vstack(wav_data_list)
    
    
    nSamples1, htkdata = load_htk(fbank_path)
    sample_per_frame = 256
    # wav = HTK.read_wav_start_end(wav_path, 0, nSamples1 * 128 + 80)
    # nSamples2 = (len(wav) - 80) // 320
    # htkdata: T * F
    # nSamples = min(nSamples1 // 2, nSamples2)
    nSamples = int((wav.shape[-1] - 80) / sample_per_frame)
    nSamples = min(nSamples1, nSamples)
    # pdb.set_trace()
    # wav = wav / 32768.
    # wav = wav.astype(np.float32)
    cur_frame, wav_fea, fbank_fea, intervals, total_frame, wav_stft_fea = 0, [], [], [], nSamples, []
    while(cur_frame < total_frame):
        if cur_frame + window_len <= total_frame:
            # print(cur_frame*sample_per_frame, (cur_frame+window_len)*sample_per_frame+80)
            wav_fea.append(wav[:, cur_frame*sample_per_frame:(cur_frame+window_len)*sample_per_frame+80])
            fbank_fea.append(htkdata[cur_frame:(cur_frame+window_len), :])
            intervals.append((cur_frame, cur_frame+window_len))
            wav_stft_fea.append(wav_fea_all_stft[:,:,cur_frame:(cur_frame+window_len),:])
            cur_frame += hop_len
        else:
            start = max(0, total_frame-window_len)
            wav_fea.append(wav[:, start*sample_per_frame:total_frame*sample_per_frame+80])
            fbank_fea.append(htkdata[start:total_frame, : ])
            intervals.append((start, total_frame))
            wav_stft_fea.append(wav_fea_all_stft[:,:,start:total_frame,:])
            cur_frame += window_len
    # pdb.set_trace()
    return wav_fea, fbank_fea, intervals, total_frame, wav_stft_fea


def load_single_channel_feature_fbank_mc_all(file_path, window_len=1024, hop_len=256, separator=1):
    # pdb.set_trace()
    wav_path, fbank_path = file_path.split()
    
    # wav_path = wav_path.replace('ch0', f'ch{str(ch)}')
    # pdb.set_trace()
    wav_data_list = []
    wav_ch0_samples = 0
    for ch in range(7):
        cur_wav_path = wav_path.replace('ch0', f'ch{str(ch)}')
        cur_wav, sr = sf.read(cur_wav_path, dtype='float32')
        if ch == 0:
            wav_ch0_samples = cur_wav.shape[0]
        cur_wav = cur_wav[:wav_ch0_samples]
        # pdb.set_trace()
        # sr, cur_wav = sio.read(cur_wav_path)
        wav_data_list.append(cur_wav)
    # pdb.set_trace()
    wav = np.vstack(wav_data_list)
    
    
    nSamples1, htkdata = load_htk(fbank_path)
    sample_per_frame = 256
    # wav = HTK.read_wav_start_end(wav_path, 0, nSamples1 * 128 + 80)
    # nSamples2 = (len(wav) - 80) // 320
    # htkdata: T * F
    # nSamples = min(nSamples1 // 2, nSamples2)
    nSamples = int((wav.shape[-1] - 80) / sample_per_frame)
    nSamples = min(nSamples1, nSamples)
    # pdb.set_trace()
    speech_mix = np.expand_dims(wav.transpose(), axis = 0)
    assert speech_mix.ndim == 3, f'expecting 3 dimensions, got {speech_mix.shape}'
    
    speech_mix = torch.from_numpy(speech_mix.astype(np.float32)).cpu()
    # pdb.set_trace()
    stft_mix = separator.stft(speech_mix)  # [B, F, T_long, Channels], complex
    batch_size, mix_freqs, mix_frames = stft_mix.shape[:3]
    # nSamples = min(nSamples1, nSamples)
    
    assert stft_mix.ndim == 4
    segment_frames = nSamples
    if mix_frames < segment_frames:
        padding_size = segment_frames - mix_frames
        # Pad the third dimension (time frames) with zeros
        # assert stft_mix.ndim == 4
        stft_mix = F.pad(stft_mix, (0, 0, 0, padding_size), mode='constant', value=0)
        # mix_frames = stft_mix.shape[2]
    elif mix_frames > segment_frames:
        stft_mix = stft_mix[:,:,:nSamples,:]
        # mix_frames = stft_mix.shape[2]
        # pdb.set_trace()
    
    batch_size, mix_freqs, mix_frames = stft_mix.shape[:3]
    print(batch_size, mix_freqs, mix_frames)
    assert mix_frames == nSamples, f'expecting mix_frames == nSamples'
    # pdb.set_trace()
    # wav = wav / 32768.
    # cur_frame, wav_fea, fbank_fea, intervals, total_frame = 0, [], [], [], nSamples
    
    # pdb.set_trace()
    return stft_mix


def load_single_channel_feature_fbank_sc_all(file_path, window_len=1024, hop_len=256, separator=1):
    # pdb.set_trace()
    wav_path, fbank_path = file_path.split()
    
    # wav_path = wav_path.replace('ch0', f'ch{str(ch)}')
    # pdb.set_trace()
    wav_data_list = []
    wav_ch0_samples = 0
    for ch in range(1):
        cur_wav_path = wav_path.replace('ch0', f'ch{str(ch)}')
        cur_wav, sr = sf.read(cur_wav_path, dtype='float32')
        if ch == 0:
            wav_ch0_samples = cur_wav.shape[0]
        cur_wav = cur_wav[:wav_ch0_samples]
        # pdb.set_trace()
        # sr, cur_wav = sio.read(cur_wav_path)
        wav_data_list.append(cur_wav)
    # pdb.set_trace()
    wav = np.vstack(wav_data_list)
    
    
    nSamples1, htkdata = load_htk(fbank_path)
    sample_per_frame = 256
    # wav = HTK.read_wav_start_end(wav_path, 0, nSamples1 * 128 + 80)
    # nSamples2 = (len(wav) - 80) // 320
    # htkdata: T * F
    # nSamples = min(nSamples1 // 2, nSamples2)
    nSamples = int((wav.shape[-1] - 80) / sample_per_frame)
    nSamples = min(nSamples1, nSamples)
    # pdb.set_trace()
    speech_mix = np.expand_dims(wav.transpose(), axis = 0)
    assert speech_mix.ndim == 3, f'expecting 3 dimensions, got {speech_mix.shape}'
    
    speech_mix = torch.from_numpy(speech_mix.astype(np.float32)).cpu()
    # pdb.set_trace()
    stft_mix = separator.stft(speech_mix)  # [B, F, T_long, Channels], complex
    batch_size, mix_freqs, mix_frames = stft_mix.shape[:3]
    # nSamples = min(nSamples1, nSamples)
    
    assert stft_mix.ndim == 4
    segment_frames = nSamples
    if mix_frames < segment_frames:
        padding_size = segment_frames - mix_frames
        # Pad the third dimension (time frames) with zeros
        # assert stft_mix.ndim == 4
        stft_mix = F.pad(stft_mix, (0, 0, 0, padding_size), mode='constant', value=0)
        # mix_frames = stft_mix.shape[2]
    elif mix_frames > segment_frames:
        stft_mix = stft_mix[:,:,:nSamples,:]
        # mix_frames = stft_mix.shape[2]
        # pdb.set_trace()
    
    batch_size, mix_freqs, mix_frames = stft_mix.shape[:3]
    print(batch_size, mix_freqs, mix_frames)
    assert mix_frames == nSamples, f'expecting mix_frames == nSamples'
    # pdb.set_trace()
    # wav = wav / 32768.
    # cur_frame, wav_fea, fbank_fea, intervals, total_frame = 0, [], [], [], nSamples
    
    # pdb.set_trace()
    return stft_mix



def process_spk_num(mask_label, speaker_embedding1):
    mask_label_sum = mask_label.sum(axis=1)
    # num_speaker = np.count_nonzero(mask_label_sum)
    # print(num_speaker)
    # continue
    speaker_indices = np.nonzero(mask_label_sum)
    # pdb.set_trace()
    
    speaker_indices_refine = [i for i in speaker_indices[0] if str(i) in speaker_embedding1.keys()]
    # print("________________________________________")
    if len(speaker_indices[0]) != len(speaker_indices_refine):
        print(speaker_indices[0], speaker_indices_refine)
    # if len(speaker_indices[0]):
    # pdb.set_trace()
    speaker_indices_refine = np.array(speaker_indices_refine, dtype=np.int64)
    mask_label = mask_label[speaker_indices_refine,:]
    # if len(speaker_indices[0]):
        # pdb.set_trace()
    # pdb.set_trace()
    speaker_embedding = speaker_embedding1
    
    return speaker_indices_refine, speaker_embedding
    # pdb.set_trace()


def load_single_channel_feature_fbank_mcwav_spknum_spk_mask(session, file_path, spk_num_rttm, spk_mask_all, speaker_embedding, window_len=1024, hop_len=256, wav_fea_all_stft=0):
    # pdb.set_trace()
    label_2classes = RTTM_to_Speaker_Mask(spk_num_rttm, max_speaker=8, frame_per_second=62.5)
    # speaker_embedding = 
    real_session = re.sub("_CH.*", "", session)
    nSamples2 = label_2classes.get_session_length(real_session)
    # mask_label, speakers = label_2classes.get_mixture_utternce_label(real_session, speaker_embedding[session], raw_session = session, start=start, end=end)
    # spk_mask_path = os.path.join(spk_mask_numpy, real_session + '.npy')
    # spk_mask_all = np.load(spk_mask_path)
    
    wav_path, fbank_path = file_path.split()
    
    # wav_path = wav_path.replace('ch0', f'ch{str(ch)}')
    # pdb.set_trace()
    wav_data_list = []
    wav_ch0_samples = 0
    for ch in range(7):
        cur_wav_path = wav_path.replace('ch0', f'ch{str(ch)}')
        sr, cur_wav = sio.read(cur_wav_path)
        
        if ch == 0:
            wav_ch0_samples = cur_wav.shape[0]
        cur_wav = cur_wav[:wav_ch0_samples]
        
        wav_data_list.append(cur_wav)
    # pdb.set_trace()
    wav = np.vstack(wav_data_list)
    
    
    nSamples1, htkdata = load_htk(fbank_path)
    sample_per_frame = 256
    # wav = HTK.read_wav_start_end(wav_path, 0, nSamples1 * 128 + 80)
    # nSamples2 = (len(wav) - 80) // 320
    # htkdata: T * F
    # nSamples = min(nSamples1 // 2, nSamples2)
    nSamples = int((wav.shape[-1] - 80) / sample_per_frame)
    # pdb.set_trace()
    nSamples = min(nSamples1, nSamples, nSamples2) # 23165 23168 23023
    # pdb.set_trace()
    # wav = wav / 32768.
    # wav = wav.astype(np.float32)
    cur_frame, wav_fea, fbank_fea, intervals, total_frame, wav_stft_fea, spk_index_list, spk_embedding_list, spk_mask_list = 0, [], [], [], nSamples, [], [], [], []
    min_spk_mask_len = 999999999
    while(cur_frame < total_frame):
        if cur_frame + window_len <= total_frame:
            # print(cur_frame*sample_per_frame, (cur_frame+window_len)*sample_per_frame+80)
            wav_fea.append(wav[:, cur_frame*sample_per_frame:(cur_frame+window_len)*sample_per_frame+80])
            fbank_fea.append(htkdata[cur_frame:(cur_frame+window_len), :])
            intervals.append((cur_frame, cur_frame+window_len))
            wav_stft_fea.append(wav_fea_all_stft[:,:,cur_frame:(cur_frame+window_len),:])
            
            mask_label, speakers = label_2classes.get_mixture_utternce_label(real_session, speaker_embedding[session], raw_session = session, start=cur_frame, end=cur_frame+window_len)
            spk_index, cur_speaker_embedding = process_spk_num(mask_label, speaker_embedding[session])
            spk_index_list.append(spk_index)
            spk_embedding_list.append(cur_speaker_embedding)
            
            cur_frame_10ms = int((cur_frame*sample_per_frame) / 160)
            end_10ms = int(((cur_frame+window_len)*sample_per_frame) / 160)
            cur_spk_mask = spk_mask_all[:, cur_frame_10ms: end_10ms]
            if min_spk_mask_len > cur_spk_mask.shape[-1]:
                min_spk_mask_len = cur_spk_mask.shape[-1]
            spk_mask_list.append(cur_spk_mask)
            # print(spk_index, cur_frame, cur_frame + window_len)
            # if not len(spk_index):
                # continue
            # pdb.set_trace()
            cur_frame += hop_len
        else:
            start = max(0, total_frame-window_len)
            wav_fea.append(wav[:, start*sample_per_frame:total_frame*sample_per_frame+80])
            fbank_fea.append(htkdata[start:total_frame, : ])
            intervals.append((start, total_frame))
            wav_stft_fea.append(wav_fea_all_stft[:,:,start:total_frame,:])
            
            mask_label, speakers = label_2classes.get_mixture_utternce_label(real_session, speaker_embedding[session], raw_session = session, start=cur_frame, end=total_frame)
            spk_index, cur_speaker_embedding = process_spk_num(mask_label, speaker_embedding[session])
            spk_index_list.append(spk_index)
            spk_embedding_list.append(cur_speaker_embedding)
            
            start_10ms = int((start*sample_per_frame) / 160)
            total_frame_10ms = int((total_frame*sample_per_frame) / 160)
            cur_spk_mask = spk_mask_all[:, start_10ms:(total_frame_10ms)]
            if min_spk_mask_len > cur_spk_mask.shape[-1]:
                min_spk_mask_len = cur_spk_mask.shape[-1]
            
            spk_mask_list.append(cur_spk_mask)

            cur_frame += window_len
    # pdb.set_trace()
    return wav_fea, fbank_fea, intervals, total_frame, wav_stft_fea, spk_index_list, spk_embedding_list, spk_mask_list, min_spk_mask_len


def load_single_channel_feature_fbank_scwav_spknum_spk_mask(session, file_path, spk_num_rttm, spk_mask_all, speaker_embedding, window_len=1024, hop_len=256, wav_fea_all_stft=0):
    # pdb.set_trace()
    label_2classes = RTTM_to_Speaker_Mask(spk_num_rttm, max_speaker=8, frame_per_second=62.5)
    # speaker_embedding = 
    real_session = re.sub("_CH.*", "", session)
    nSamples2 = label_2classes.get_session_length(real_session)
    # mask_label, speakers = label_2classes.get_mixture_utternce_label(real_session, speaker_embedding[session], raw_session = session, start=start, end=end)
    # spk_mask_path = os.path.join(spk_mask_numpy, real_session + '.npy')
    # spk_mask_all = np.load(spk_mask_path)
    
    wav_path, fbank_path = file_path.split()
    
    # wav_path = wav_path.replace('ch0', f'ch{str(ch)}')
    # pdb.set_trace()
    wav_data_list = []
    wav_ch0_samples = 0
    for ch in range(1):
        cur_wav_path = wav_path.replace('ch0', f'ch{str(ch)}')
        sr, cur_wav = sio.read(cur_wav_path)
        
        if ch == 0:
            wav_ch0_samples = cur_wav.shape[0]
        cur_wav = cur_wav[:wav_ch0_samples]
        
        wav_data_list.append(cur_wav)
    # pdb.set_trace()
    wav = np.vstack(wav_data_list)
    
    
    nSamples1, htkdata = load_htk(fbank_path)
    sample_per_frame = 256
    # wav = HTK.read_wav_start_end(wav_path, 0, nSamples1 * 128 + 80)
    # nSamples2 = (len(wav) - 80) // 320
    # htkdata: T * F
    # nSamples = min(nSamples1 // 2, nSamples2)
    nSamples = int((wav.shape[-1] - 80) / sample_per_frame)
    # pdb.set_trace()
    nSamples = min(nSamples1, nSamples, nSamples2) # 23165 23168 23023
    # pdb.set_trace()
    # wav = wav / 32768.
    # wav = wav.astype(np.float32)
    cur_frame, wav_fea, fbank_fea, intervals, total_frame, wav_stft_fea, spk_index_list, spk_embedding_list, spk_mask_list = 0, [], [], [], nSamples, [], [], [], []
    min_spk_mask_len = 999999999
    while(cur_frame < total_frame):
        if cur_frame + window_len <= total_frame:
            # print(cur_frame*sample_per_frame, (cur_frame+window_len)*sample_per_frame+80)
            wav_fea.append(wav[:, cur_frame*sample_per_frame:(cur_frame+window_len)*sample_per_frame+80])
            fbank_fea.append(htkdata[cur_frame:(cur_frame+window_len), :])
            intervals.append((cur_frame, cur_frame+window_len))
            wav_stft_fea.append(wav_fea_all_stft[:,:,cur_frame:(cur_frame+window_len),:])
            
            mask_label, speakers = label_2classes.get_mixture_utternce_label(real_session, speaker_embedding[session], raw_session = session, start=cur_frame, end=cur_frame+window_len)
            spk_index, cur_speaker_embedding = process_spk_num(mask_label, speaker_embedding[session])
            spk_index_list.append(spk_index)
            spk_embedding_list.append(cur_speaker_embedding)
            
            cur_frame_10ms = int((cur_frame*sample_per_frame) / 160)
            end_10ms = int(((cur_frame+window_len)*sample_per_frame) / 160)
            cur_spk_mask = spk_mask_all[:, cur_frame_10ms: end_10ms]
            if min_spk_mask_len > cur_spk_mask.shape[-1]:
                min_spk_mask_len = cur_spk_mask.shape[-1]
            spk_mask_list.append(cur_spk_mask)
            # print(spk_index, cur_frame, cur_frame + window_len)
            # if not len(spk_index):
                # continue
            # pdb.set_trace()
            cur_frame += hop_len
        else:
            start = max(0, total_frame-window_len)
            wav_fea.append(wav[:, start*sample_per_frame:total_frame*sample_per_frame+80])
            fbank_fea.append(htkdata[start:total_frame, : ])
            intervals.append((start, total_frame))
            wav_stft_fea.append(wav_fea_all_stft[:,:,start:total_frame,:])
            
            mask_label, speakers = label_2classes.get_mixture_utternce_label(real_session, speaker_embedding[session], raw_session = session, start=cur_frame, end=total_frame)
            spk_index, cur_speaker_embedding = process_spk_num(mask_label, speaker_embedding[session])
            spk_index_list.append(spk_index)
            spk_embedding_list.append(cur_speaker_embedding)
            
            start_10ms = int((start*sample_per_frame) / 160)
            total_frame_10ms = int((total_frame*sample_per_frame) / 160)
            cur_spk_mask = spk_mask_all[:, start_10ms:(total_frame_10ms)]
            if min_spk_mask_len > cur_spk_mask.shape[-1]:
                min_spk_mask_len = cur_spk_mask.shape[-1]
            
            spk_mask_list.append(cur_spk_mask)

            cur_frame += window_len
    # pdb.set_trace()
    return wav_fea, fbank_fea, intervals, total_frame, wav_stft_fea, spk_index_list, spk_embedding_list, spk_mask_list, min_spk_mask_len




def preds_to_rttm(preds, intervals, total_dur, output_path):
    # pdb.set_trace()
    rttm = np.zeros([preds[0].shape[0], total_dur]) #rttm是以session为单位， shape: num_speaker * dur, preds是同一个session的预测结果，每个元素是num_speaker * cur_utt_durance
    weight = np.zeros(total_dur)
    for i, p in enumerate(preds):
        rttm[:, intervals[i][0]: intervals[i][1] - 2] += p
        weight[ intervals[i][0]: intervals[i][1] - 2] += 1         #累计重叠次数。因为滑窗是有重叠滑的原因，有些帧会被重复计算，所以需要除以weight
    np.save(output_path, rttm / (weight[None, :] + np.finfo(float).eps))


def preds_to_wav_joint_train(mask_preds, sep_preds, window_spks, num_speaker, new_intervals, total_frame):
    # pdb.set_trace()
    B = len(window_spks)
    _, F, T, _ = sep_preds[0].shape
    mask_preds_all = sep_preds[0].new_zeros(1, F, total_frame, num_speaker, dtype=torch.float)
    sep_preds_all = sep_preds[0].new_zeros(1,F, total_frame, num_speaker)
    weight = sep_preds[0].new_zeros(total_frame, num_speaker, dtype=torch.float)
    
    
    for i in range(B):
        # pdb.set_trace()
        mask_preds_all[:, :, new_intervals[i][0]:new_intervals[i][0] + T,  window_spks[i] ] += mask_preds[i][...,: len(window_spks[i])]
        sep_preds_all[:, :, new_intervals[i][0]:new_intervals[i][0] + T,  window_spks[i] ] += sep_preds[i][...,: len(window_spks[i])]
        weight[new_intervals[i][0]: new_intervals[i][0] + T, window_spks[i]] += 1         #累计重叠次数。因为滑窗是有重叠滑的原因，有些帧会被重复计算，所以需要除以weight
    # pdb.set_trace()
    mask_preds_all /= (weight[None, None, :, :] + np.finfo(float).eps)
    sep_preds_all /= (weight[None, None, :, :] + np.finfo(float).eps)
    # pdb.set_trace()
    return sep_preds_all.squeeze(), mask_preds_all.squeeze()
    # np.save(output_path, rttm / (weight[None, :] + np.finfo(float).eps))

def preds_to_mask_joint_train(mask_preds, noise_preds, window_spks, num_speaker, new_intervals, total_frame):
    # pdb.set_trace()
    B = len(window_spks)
    _, F, T, _ = mask_preds[0].shape
    mask_preds_all = mask_preds[0].new_zeros(1, F, total_frame, num_speaker, dtype=torch.float)
    noise_preds_all = mask_preds[0].new_zeros(1,F, total_frame, 1, dtype=torch.float)
    
    weight = mask_preds[0].new_zeros(total_frame, num_speaker, dtype=torch.float)
    weight_noise = mask_preds[0].new_zeros(total_frame, dtype=torch.float)
    
    
    for i in range(B):
        # pdb.set_trace()
        mask_preds_all[:, :, new_intervals[i][0]:new_intervals[i][0] + T,  window_spks[i] ] += mask_preds[i][...,: len(window_spks[i])]
        noise_preds_all[:, :, new_intervals[i][0]:new_intervals[i][0] + T,  :] += noise_preds[i]
        weight[new_intervals[i][0]: new_intervals[i][0] + T, window_spks[i]] += 1         #累计重叠次数。因为滑窗是有重叠滑的原因，有些帧会被重复计算，所以需要除以weight
        weight_noise[new_intervals[i][0]: new_intervals[i][0] + T] += 1 
    # pdb.set_trace()
    mask_preds_all /= (weight[None, None, :, :] + np.finfo(float).eps)
    noise_preds_all /= (weight_noise[None, None, :, None] + np.finfo(float).eps)
    # pdb.set_trace()
    return noise_preds_all.squeeze(), mask_preds_all.squeeze()

def switch_wav_pipline(mask_preds, sep_preds, new_intervals):
    masked_seg_list = sep_preds[:]
    spk_masks_list = mask_preds[:]
    total_frame = new_intervals[-1][-1]
    B, F, T, num_spks = sep_preds[0].shape
    segment_frames = new_intervals[0][1] - new_intervals[0][0] - 2
    m0_frames = int(0.15 / 0.016)
    m1_frames = int(0.3 / 0.016)
    # pdb.set_trace()
    # stitch the separated segments together
    stft_stitched = sep_preds[0].new_zeros(1,F, total_frame, num_spks)   # [B, F, T_long, num_spks]
    mask_stitched = sep_preds[0].new_zeros(1,F, total_frame, num_spks, dtype=torch.float)
    wg_stitched = sep_preds[0].new_zeros(total_frame, dtype=torch.float32)  # [T_long]
    # add first segment
    wg_seg = calc_segment_weight(segment_frames, m0_frames, m1_frames, is_first_seg=True)
    # pdb.set_trace()
    wg_seg = wg_seg.cuda()
    wg_stitched[:segment_frames] += wg_seg
    stft_stitched[:, :, :segment_frames] += wg_seg.view(1, 1, -1, 1) * masked_seg_list[0]
    mask_stitched[:, :, :segment_frames] += wg_seg.view(1, 1, -1, 1) * spk_masks_list[0]
    # pdb.set_trace()
    pit = PitWrapper({'mse': mse_loss, 'l1': l1_loss}['l1'])
    num_segments = len(spk_masks_list)
    # II. stitch the separated segments together
    mask_flag = 1
    masked_mag_flag = 0
    for i in range(1, num_segments):
        if mask_flag:
            left_input, right_input = spk_masks_list[i-1], spk_masks_list[i]
        elif masked_mag_flag:
            # masked magnitudes
            left_input, right_input = masked_seg_list[i - 1].abs(), masked_seg_list[i].abs()
        else:
            assert False, f'unexpected stitching_input: {cfg.stitching_input}'
        # pdb.set_trace()
        overlap_frames = new_intervals[i-1][1] - new_intervals[i][0] - 2
        assert left_input.shape[2] == right_input.shape[2] == segment_frames
        loss, right_perm = pit(left_input[:, :, -overlap_frames:], right_input[:, :, :overlap_frames])

        # Plot for debugging:
        # plot_left_right_stitch(separator, left_input, right_input, right_perm,
        #                        overlap_frames, cfg, stft_seg_to_play=masked_seg_list[i][..., 0], fs=fs)

        # permute current segment to match with the previous one
        for ib in range(B):
            spk_masks_list[i][ib] = spk_masks_list[i][ib, ..., right_perm[ib]]
            masked_seg_list[i][ib] = masked_seg_list[i][ib, ..., right_perm[ib]]
        if i != num_segments-1:
            st = new_intervals[i][0]
            en = new_intervals[i][1] - 2
        else:
            st = new_intervals[i][0]
            en = new_intervals[i][1]
            segment_frames += 2
        # weighted overlap-and-add
        wg_seg = calc_segment_weight(segment_frames, m0_frames, m1_frames, is_last_seg=(i==num_segments-1))
        wg_seg = wg_seg[:en-st].cuda()  # last segment may be shorter
        wg_stitched[st:en] += wg_seg
        assert torch.is_complex(masked_seg_list[i]), 'summation assumes complex representation'
        if i == num_segments-1:
            en -= 2
            wg_seg = wg_seg[:-2]
        stft_stitched[:, :, st:en] += wg_seg.view(1, 1, -1, 1) * masked_seg_list[i][:, :, :en-st]
        mask_stitched[:, :, st:en] += wg_seg.view(1, 1, -1, 1) * spk_masks_list[i][:, :, :en-st]
        # pdb.set_trace()
    # pdb.set_trace()
    assert (wg_stitched > 1e-5).all(), 'zero weights found. check hop_size, segment_size or m0, m1'
    stft_stitched /= wg_stitched.view(1, 1, -1, 1) # stft_stitched torch.Size([1, 257, 22627, 3])
    mask_stitched /= wg_stitched.view(1, 1, -1, 1) # 
    # pdb.set_trace()
    return stft_stitched.squeeze(), mask_stitched.squeeze()


def switch_wav_pipline_all(mask_preds, sep_preds, new_intervals, separator):
    masked_seg_list = sep_preds[:]
    spk_masks_list = mask_preds[:]
    total_frame = new_intervals[-1][-1]
    B, F, T, num_spks = sep_preds[0].shape
    segment_frames = new_intervals[0][1] - new_intervals[0][0]
    m0_frames = int(0.15 / 0.016)
    m1_frames = int(0.3 / 0.016)
    # pdb.set_trace()
    # stitch the separated segments together
    stft_stitched = sep_preds[0].new_zeros(1,F, total_frame, num_spks)   # [B, F, T_long, num_spks]
    mask_stitched = sep_preds[0].new_zeros(1,F, total_frame, num_spks, dtype=torch.float)
    wg_stitched = sep_preds[0].new_zeros(total_frame, dtype=torch.float32)  # [T_long]
    # add first segment
    wg_seg = calc_segment_weight(segment_frames, m0_frames, m1_frames, is_first_seg=True)
    # pdb.set_trace()
    wg_seg = wg_seg.cuda()
    wg_stitched[:segment_frames] += wg_seg
    stft_stitched[:, :, :segment_frames] += wg_seg.view(1, 1, -1, 1) * masked_seg_list[0]
    mask_stitched[:, :, :segment_frames] += wg_seg.view(1, 1, -1, 1) * spk_masks_list[0]
    # pdb.set_trace()
    pit = PitWrapper({'mse': mse_loss, 'l1': l1_loss}['l1'])
    num_segments = len(spk_masks_list)
    # II. stitch the separated segments together
    mask_flag = 1
    masked_mag_flag = 0
    for i in range(1, num_segments):
        if mask_flag:
            left_input, right_input = spk_masks_list[i-1], spk_masks_list[i]
        elif masked_mag_flag:
            # masked magnitudes
            left_input, right_input = masked_seg_list[i - 1].abs(), masked_seg_list[i].abs()
        else:
            assert False, f'unexpected stitching_input: {cfg.stitching_input}'
        # pdb.set_trace()
        overlap_frames = new_intervals[i-1][1] - new_intervals[i][0]
        assert left_input.shape[2] == right_input.shape[2] == segment_frames
        loss, right_perm = pit(left_input[:, :, -overlap_frames:], right_input[:, :, :overlap_frames])

        # Plot for debugging:
        # pdb.set_trace()
        # plot_left_right_stitch(separator, left_input.cpu(), right_input.cpu(), right_perm,
                               # overlap_frames, num_spks = 3, stft_seg_to_play=masked_seg_list[i][..., 0].cpu(), fs=16000)
        # pdb.set_trace()
        # plot_separation_methods(stft_seg_device_chref, masks, mvdr_responses, separator, num_spks = 3,
                         # plots=['mvdr', 'masked_mvdr', 'spk_masks', 'masked_ref_ch', 'mixture'])
        
        # permute current segment to match with the previous one
        for ib in range(B):
            spk_masks_list[i][ib] = spk_masks_list[i][ib, ..., right_perm[ib]]
            masked_seg_list[i][ib] = masked_seg_list[i][ib, ..., right_perm[ib]]
        if i != num_segments-1:
            st = new_intervals[i][0]
            en = new_intervals[i][1]
        else:
            st = new_intervals[i][0]
            en = new_intervals[i][1]
            # segment_frames += 2
        # weighted overlap-and-add
        wg_seg = calc_segment_weight(segment_frames, m0_frames, m1_frames, is_last_seg=(i==num_segments-1))
        wg_seg = wg_seg[:en-st].cuda()  # last segment may be shorter
        wg_stitched[st:en] += wg_seg
        assert torch.is_complex(masked_seg_list[i]), 'summation assumes complex representation'
        # if i == num_segments-1:
            # en -= 2
            # wg_seg = wg_seg[:-2]
        stft_stitched[:, :, st:en] += wg_seg.view(1, 1, -1, 1) * masked_seg_list[i][:, :, :en-st]
        mask_stitched[:, :, st:en] += wg_seg.view(1, 1, -1, 1) * spk_masks_list[i][:, :, :en-st]
        # pdb.set_trace()
    # pdb.set_trace()
    assert (wg_stitched > 1e-5).all(), 'zero weights found. check hop_size, segment_size or m0, m1'
    stft_stitched /= wg_stitched.view(1, 1, -1, 1) # stft_stitched torch.Size([1, 257, 22627, 3])
    mask_stitched /= wg_stitched.view(1, 1, -1, 1) # 
    # pdb.set_trace()
    return stft_stitched.squeeze(), mask_stitched.squeeze()




def calc_segment_weight(seg_frames: int, m0_frames: int, m1_frames: int,
                        is_first_seg: bool = False, is_last_seg: bool = False):

    """
    Returns weighting for segment.

    During weighted overlap-and-add the separated segments will be weighted by a time-window defined
    by m0 and m1 parameters.
    Frames 0 to m0_frames will have weight=0, m1_frames and onward will have weight=1.
    Frames between m0_frames and m1_frames will have linearly increasing weight.
    The weights on the right side will behave symetrically.

        Weight
    1     |            ____________
          |           /            \
          |          /              \
          |         /                \
    0     |________/                  \________
          0      m0  m1           m1' m0'
                 <---->           <---->
             Linear Increase   Linear Decrease

    m1' = seg_frames - m1
    m0' = seg_frames - m0


    Args:
        seg_frames: segment length in frames
        m0_frames: start of linear increase
        m1_frames: end of linear increase
        is_first_seg: True if this is the first segment in the long-form audio
        is_last_seg: True if this is the last segment in the long-form audio
    """
    assert seg_frames > 2 * m1_frames, \
        'not enough frames to fit weighting window. try modifying hop_size, segment_size or m0, m1'
    wg_win = torch.ones(seg_frames, dtype=torch.float32)
    wg_win[:m0_frames] = 0
    wg_win[len(wg_win)-m0_frames:] = 0
    linear = torch.linspace(0.1, 1, m1_frames - m0_frames)  # linear transition from 0.1 to 1
    wg_win[m0_frames:m1_frames] = linear
    wg_win[-m1_frames:-m0_frames] = torch.flip(linear, (0,))

    if is_first_seg:
        # first segment is the only contributor to its left edge, so we can't have zero weight.
        wg_win[:m0_frames] = 0.1
    if is_last_seg:
        # similar to the above, last segment is the only contirbutor to its right edge.
        wg_win[len(wg_win) - m0_frames:] = 0.1

    return wg_win








def get_rttm(input_path, wav_data, output_path, session_cur):
    # rttm = {}
    with open(input_path, 'r') as file:
        lines = file.readlines()
    file_speaker_ids = {}
    for line in lines:
        parts = line.strip().split()
        file_name = parts[1]
        speaker_id = parts[7]
        
        if file_name not in file_speaker_ids:
            file_speaker_ids[file_name] = []
        
        if speaker_id not in file_speaker_ids[file_name]:
            file_speaker_ids[file_name].append(speaker_id)
    
    # 为每个文件名创建新的ID映射和反向映射
    file_id_mappings = {}
    file_reverse_mappings = {}
    
    for file_name, speaker_ids in file_speaker_ids.items():
        id_mapping = {old_id: str(new_id) for new_id, old_id in enumerate(speaker_ids)}
        reverse_mapping = {str(new_id): old_id for new_id, old_id in enumerate(speaker_ids)}
        file_id_mappings[file_name] = id_mapping
        file_reverse_mappings[file_name] = reverse_mapping
    
    
    
    T = 180 * 60 * 100
    with open(input_path) as INPUT:
        for line in INPUT:
            '''
            SPEAKER session0_CH0_0S 1 417.315   9.000 <NA> <NA> 1 <NA> <NA>
            '''
            line = line.split(" ")
            while "" in line:
                line.remove("")
            session = line[1]
            if session != session_cur:
                continue
            # if not session in rttm.keys() :
                # rttm[session] = {}
            if line[-2] != "<NA>":
                spk = line[-2]
            else:
                spk = line[-3]
            # spk = file_id_mappings[session][spk]
            # pdb.set_trace()
            # if not spk in rttm[session].keys():
                # rttm[session][spk] = np.zeros(T)
            #print(line[3] )
            # pdb.set_trace()
            if np.int64(np.float64(line[4]) * 100) < 20:
                continue
            start_name = np.int64(np.float64(line[3]) * 100 )
            end_name = start_name + np.int64(np.float64(line[4]) * 100)
            
            start_wav = np.int64(np.float64(line[3]) * 16000 )
            end_wav = start_wav + np.int64(np.float64(line[4]) * 16000 )
            cur_wav = wav_data[int(spk), start_wav:end_wav] # [N, T]
            out_name = '_'.join([spk, session, "%06d" % start_name + '-' + "%06d" % end_name])
            write_wav(output_path + f'/{out_name}.wav' ,samps=cur_wav, sr=16000)
            # pdb.set_trace()
            # rttm[session][spk][start:end] = 1
    # return rttm


def main(args):
    pit = PitWrapper({'mse': mse_loss, 'l1': l1_loss}['l1'])
    sep_models_dir = '/train33/sppro/permanent/stniu/NOTSOFAR1-Challenge-main/artifacts/css_models/notsofar/conformer1.0_decode'
    separator, _ = load_css_model(Path(sep_models_dir) / 'mc')
    separator.cpu()
    separator.eval()
    
    # if not os.path.exists(args.output_dir):
        # os.makedirs(args.output_dir, exist_ok = True)
    #torch.cuda.set_device(0)
    print("model type is {}".format(args.model_type))
    
    model = importlib.import_module('.', package='{}'.format(args.model_type))
    
    try:
        if args.model_config != "N":
            print(config.configs[args.model_config])
            nnet = model.MULTI_MAM_SE_S2S_model(config.configs[args.model_config])
        else:
            configs = torch.load(args.model_path)["configs"]
            print("configs from model ", configs)
            nnet = model.MULTI_MAM_SE_S2S_model(configs)
    except:
        #nnet = SE_MA_MSE_NSD(config.configs[args.model_config])
        raise "load model error!!!"
    # pdb.set_trace()
    utils_s2s.load_checkpoint_join_training_init_mask_decode(nnet, None, args.model_path)
    
    
    
    gpus = list(range(len(args.gpu.split(','))))
    nnet = torch.nn.DataParallel(nnet, device_ids = gpus).cuda()
    
    
    nnet.eval()
    # nnet_blind.eval()
    # nnet_fix.eval()
    Sigmoid = torch.nn.Sigmoid()  # 转换成概率

    file_list = {}
    with open(args.feature_list) as INPUT:
        for l in INPUT:
            session = os.path.basename(l).split('.')[0]
            file_list[session] = l.rstrip()

    spk_mask_numpy = './data/MTG1_eval_sc/MULTI_MAM_SE_S2S_model.model10_MTG_eval_sc_sq_0711_fusion'
    
    print(args.embedding_list)
    embedding, _ = load_ivector(args.embedding_list)
    _, train_set_speaker_embedding = load_ivector(args.train_set_speaker_embedding_list)
    idxs = list(range(len(train_set_speaker_embedding)))
    list_out = []
    for session in tqdm.tqdm(file_list.keys()):
        print(f'---------------{session}---------------')
        # if session.replace('-mc','').replace('_CH0', '') != 'MTG_30830_plaza_0':
            # continue
            
        real_session = re.sub("_CH.*", "", session)
        speaker_embedding = []
        speaker_list = list(embedding[session].keys())
        num_speaker = len(speaker_list)
        
        
        spk_num_rttm = './data/MTG1_eval_sc/MULTI_MAM_SE_S2S_model.model10_MTG_eval_sc_sq_0711_fusion/rttm_th0.50_pp_split'
        
        print(spk_mask_numpy)
        print(spk_num_rttm)
        # pdb.set_trace()
        label_init = RTTM_to_Speaker_Mask(spk_num_rttm,  max_speaker=args.max_speaker, frame_per_second=args.frame_per_second)
        speaker_list1 = list(label_init.frame_label[real_session].keys())
        # print(speaker_list, speaker_list1)
        
        if speaker_list != speaker_list1:
            list_out.append([session, speaker_list, speaker_list1, len(speaker_list) - len(speaker_list1)])
            if len(speaker_list) - len(speaker_list1) == 0:
                new_dict1 = {}
                new_keys = [str(ni) for ni in range(len(speaker_list1))]
                for key_index, key in enumerate(speaker_list):
                    # pdb.set_trace()
                    new_dict1[new_keys[key_index]] = 0
                    new_dict1[new_keys[key_index]] = embedding[session][key]
                
                # print(session, speaker_list, speaker_list1)
                # print(session, new_dict1.keys(), speaker_list1)
                embedding[session] = new_dict1
                speaker_list1 = new_keys
            else:
                pdb.set_trace()
                new_dict1 = {}
                new_dict2 = {}
                new_keys = [str(ni) for ni in range(len(speaker_list1))]
                common_keys = sorted(list(set(speaker_list).intersection(set(speaker_list1))))
                for key_index, key in enumerate(common_keys):
                    new_dict1[new_keys[key_index]] = 0
                    new_dict1[new_keys[key_index]] = embedding[session][key]
                
                for key_index, key in enumerate(common_keys):
                    new_dict2[new_keys[key_index]] = 0
                    new_dict2[new_keys[key_index]] = label_init.frame_label[real_session][key]
                list_out.append([session, speaker_list, speaker_list1, len(speaker_list) - len(speaker_list1)])

                embedding[session] = new_dict1
                label_init.frame_label[real_session] = new_dict2
                speaker_list1 = new_keys
        # else:
            # continue
                # pdb.set_trace()
        # print(list_out)
        for item_list_out in list_out:
            print(item_list_out)
        print(speaker_list, speaker_list1)
        # continue
        speaker_list = speaker_list1
        num_speaker = len(speaker_list)
        num_speaker = int(max(speaker_list)) + 1
        if num_speaker > args.max_speaker: print(speaker_list)
        for spk in embedding[session].keys():
            speaker_embedding.append(embedding[session][spk])
        for idx in np.random.choice(idxs, args.max_speaker - num_speaker, replace=False):
            speaker_embedding.append(train_set_speaker_embedding[idx])
        speaker_embedding = torch.stack(speaker_embedding) # num_speaker * embedding_dim
        # pdb.set_trace()
        spk_mask_path = os.path.join(spk_mask_numpy, real_session + '.npy')
        spk_mask_all = np.load(spk_mask_path)
        # pdb.set_trace()
        wav_fea_all_stft = load_single_channel_feature_fbank_sc_all(file_list[session], args.max_utt_durance, args.hop_len, separator)
        # spk_mask_all = np.zeros([num_speaker, int(wav_fea_all_stft.shape[-2] * 1.7)])
        wav_fea, feature, intervals, total_frame, wav_stft_fea, spk_index_list, spk_embedding_list, spk_mask_list, min_spk_mask_len = load_single_channel_feature_fbank_scwav_spknum_spk_mask(session, file_list[session], spk_num_rttm, spk_mask_all, embedding, args.max_utt_durance, args.hop_len, wav_fea_all_stft)

        preds, i, cur_utt, batch, batch_wav, batch_intervals, new_intervals, batch_wav_stft_fea, batch_spk_index_list, batch_spk_embedding_list, batch_spk_mask_list = [], 0, 0, [], [], [], [], [], [], [], []
        sep_preds = []
        mask_preds = []
        noise_preds = []
        window_spks = []
        window_max_spk = 4
        # batc_list = []
        # pdb.set_trace()
        with torch.no_grad():
            for m in feature:
                cur_wav_fea = wav_fea[cur_utt]
                cur_wav_seft_fea = wav_stft_fea[cur_utt]
                cur_spk_index = spk_index_list[cur_utt]
                cur_speaker_embedding1 = spk_embedding_list[cur_utt]
                cur_spk_mask1 = spk_mask_list[cur_utt]
                
                batch_wav.append(torch.from_numpy(cur_wav_fea.astype(np.float32)))
                batch.append(torch.from_numpy(m.astype(np.float32)))
                batch_intervals.append(intervals[cur_utt])
                batch_wav_stft_fea.append(cur_wav_seft_fea)
                
                
                # pdb.set_trace()
                cur_numspk = len(cur_spk_index)
                # print('old: ', cur_spk_index)
                
                if cur_numspk > window_max_spk:
                    # pdb.set_trace()
                    mask_temp = label_init.get_mixture_utternce_label_informed_speaker(real_session, speaker_list, start=intervals[cur_utt][0], end=intervals[cur_utt][1], max_speaker=args.max_speaker)
                    cur_spk_index = np.argsort(mask_temp.sum(axis=-1))[-window_max_spk:]
                    # cur_spk_index = np.argsort(cur_spk_mask1.sum(axis=-1))[-window_max_spk:]
                    cur_numspk = len(cur_spk_index)
                    print('new: ', cur_spk_index)

                
                cur_speaker_embedding1 = [cur_speaker_embedding1[str(i)] for i in cur_spk_index]
                assert len(cur_spk_index) == len(cur_speaker_embedding1)
                
                for idx in np.random.choice(idxs, window_max_spk - cur_numspk, replace=False):
                    cur_speaker_embedding1.append(train_set_speaker_embedding[idx])
                cur_speaker_embedding1 = torch.stack(cur_speaker_embedding1)
                batch_spk_embedding_list.append(cur_speaker_embedding1)
                
                
                
                cur_spk_mask1 = cur_spk_mask1[cur_spk_index,:min_spk_mask_len]
                if cur_numspk < window_max_spk:
                    append_spk_mask = np.zeros([window_max_spk - cur_numspk, min_spk_mask_len])
                    cur_spk_mask1 = np.vstack([cur_spk_mask1, append_spk_mask])
                
                
                batch_spk_mask_list.append(torch.from_numpy(cur_spk_mask1.astype(np.float32)))
                batch_spk_index_list.append(cur_spk_index)
                
                
                cur_utt += 1
                i += 1
                if (i == args.batch_size) or (len(feature) == cur_utt): #攒够一个batch
                    # print(f'----------{str(cur_utt)}----------')
                    length = [item.shape[0] for item in batch]
                    ordered_index = sorted(range(len(length)), key=lambda k: length[k], reverse = True)
                    # pdb.set_trace()
                    CH, wav_Time = batch_wav[ordered_index[0]].shape
                    Time, Freq = batch[ordered_index[0]].shape
                    cur_batch_size = len(length)
                    wav_data = np.zeros([cur_batch_size, CH, wav_Time]).astype(np.float32)
                    input_data = np.zeros([cur_batch_size, Time, Freq]).astype(np.float32)
                    mask_data = np.zeros([cur_batch_size, window_max_spk, Time]).astype(np.float32)
                    # pdb.set_trace()
                    wav_data_stft = cur_wav_seft_fea.new_zeros([cur_batch_size, 257, Time, CH])
                    spk_masks = np.zeros([cur_batch_size, window_max_spk, min_spk_mask_len]).astype(np.float32)
                    
                    
                    
                    nframes = []
                    batch_speaker_embedding = []
                    batch_speaker_embedding1 = []
                    batch_spk_index = []
                    for i, id in enumerate(ordered_index):
                        # pdb.set_trace()
                        # print(batch_intervals)
                        wav_data[i, :, :length[id]*256+80] = batch_wav[id]
                        input_data[i, :length[id], :] = batch[id]
                        wav_data_stft[i, :, :length[id], :] = batch_wav_stft_fea[id]
                        nframes.append(length[id])
                        batch_speaker_embedding.append(speaker_embedding)
                        batch_speaker_embedding1.append(batch_spk_embedding_list[id])
                        batch_spk_index.append(batch_spk_index_list[id])
                        spk_masks[i, :, :] = batch_spk_mask_list[id]
                        
                        
                        mask = label_init.get_mixture_utternce_label_informed_speaker(real_session, speaker_list, start=batch_intervals[id][0], end=batch_intervals[id][1], max_speaker=args.max_speaker)
                        if args.remove_overlap: #为什么要去掉重叠的部分？ 因为MAMSE只需要特定说话人的mask
                            overlap = np.sum(mask, axis=0)
                            mask[:, overlap>=2] = 0
                            # pdb.set_trace()
                        mask = mask[batch_spk_index_list[id],:]
                        if mask.shape[0] < window_max_spk:
                            append_mask = np.zeros([window_max_spk - mask.shape[0], length[id]])
                            mask = np.vstack([mask, append_mask])
                            
                        mask_data[i, :, :mask.shape[1]] = mask
                        new_intervals.append(batch_intervals[id])
                        
                    wav_data = torch.from_numpy(wav_data).cuda()
                    input_data = torch.from_numpy(input_data).transpose(1, 2).cuda()
                    wav_data_stft = wav_data_stft.cuda()
                    spk_masks = torch.from_numpy(spk_masks).cuda()
                    #print("input", input_data.shape)
                    batch_speaker_embedding = torch.stack(batch_speaker_embedding).cuda() # B * num_speaker * embedding_dim
                    batch_speaker_embedding1 = torch.stack(batch_speaker_embedding1).cuda()
                    
                    mask_data = torch.from_numpy(mask_data).cuda()
                    # pdb.set_trace()
                    ypreds_list = []
                    noise_mask_list = []
                    for ch in range(1):
                        # pdb.set_trace()
                        spk_masks_reshape = F.interpolate(spk_masks, size=(800,), mode='nearest')
                        dia_ypreds1, dia_ypreds2, ypreds, noise_mask, css_model = nnet(wav_data[:,ch:ch+1,:], spk_masks_reshape, wav=1)
                        ypreds1 = ypreds
                        noise_mask1 = noise_mask
                        
                        ypreds_list.append(ypreds1)
                        noise_mask_list.append(noise_mask1)
                    ypreds = sum(ypreds_list) / len(ypreds_list)
                    noise_mask = sum(noise_mask_list) / len(noise_mask_list)
                    
                    
                    data_wav = wav_data[:,:,:-81]
                    data_wav = data_wav.permute(0,2,1)
                    
                    mc = 0
                    if mc:
                        mix_mic0_mag_ft_mc = css_model.stft(data_wav[:, :, :]) # [12, 257, 198, 7]
                        bs = mix_mic0_mag_ft_mc.shape[0]
                        seg_for_masking_list = []
                        for i in range(bs):
                            mvdr_responses = make_mvdr(ypreds[i].squeeze(0).moveaxis(2, 0).cpu().numpy(),
                                   noise_mask[i].squeeze(0).moveaxis(2, 0).cpu().numpy(),
                                   mix_stft=mix_mic0_mag_ft_mc[i].squeeze(0).moveaxis(2, 0).cpu().numpy(),
                                   return_stft=True)
                            mvdr_responses = torch.from_numpy(np.stack(mvdr_responses, axis=-1)).unsqueeze(0).cuda()
                            seg_for_masking = mvdr_responses
                            seg_for_masking_list.append(seg_for_masking)
                        seg_for_masking_out = torch.vstack(seg_for_masking_list)
                        mask_floor_db = 0
                        mask_floor = 10. ** (mask_floor_db / 20.)
                        mask_clipped = torch.clip(ypreds, min=mask_floor)
                        sep_out = seg_for_masking_out * mask_clipped
                    
                    sc = 1
                    if sc:
                        ref_mic = 0
                        mix_mic0_mag_ft = css_model.stft(data_wav[:, :, ref_mic])[..., None]
                        mask_floor_db = -np.inf
                        mask_floor = 10. ** (mask_floor_db / 20.)
                        mask_clipped = torch.clip(ypreds, min=mask_floor)
                        
                        sep_out = mask_clipped * mix_mic0_mag_ft
                    # pdb.set_trace()
                    
                    # mix_mic0_mag_ft = mix_mic0_mag_ft[..., 0]
                    if 0:
                            # normalize to match the input mixture power
                        assert mix_mic0_mag_ft.ndim == 3
                        mix_energy = torch.sqrt(
                            torch.mean(mix_mic0_mag_ft[:, :, :].abs().pow(2),  # squared mag
                                       dim=(1, 2), keepdim=True)
                        )

                        assert torch.is_complex(sep_out)
                        sep_energy = torch.sqrt(
                            torch.mean(sep_out[:, :, :, :].sum(-1).abs().pow(2),  # sum over spks, squared mag
                                       dim=(1, 2), keepdim=True
                            )
                        )
                        # pdb.set_trace()
                        sep_out = (mix_energy / sep_energy)[..., None]  * sep_out
                    
                    
                    # pdb.set_trace()
                    
                    for k in range(ypreds.shape[0]):
                        # pdb.set_trace()
                        mask_preds.append(ypreds[k].unsqueeze(dim = 0))
                        sep_preds.append(sep_out[k].unsqueeze(dim = 0))
                        noise_preds.append(noise_mask[k].unsqueeze(dim = 0))
                        window_spks.append(batch_spk_index[k])
                    # pdb.set_trace()
                    
                    

                    i, batch, batch_intervals, batch_wav, batch_wav_stft_fea, batch_spk_index_list, batch_spk_embedding_list, batch_spk_mask_list = 0, [], [], [], [], [], [], []
                    # pdb.set_trace()
                    # i, batch, batch_intervals = 0, [], []
        # pdb.set_trace()
        # out_noise_mask, out_spk_mask = preds_to_mask_joint_train(mask_preds, noise_preds, window_spks, num_speaker, new_intervals, total_frame)
        # out_mask = torch.cat([out_spk_mask, out_noise_mask.unsqueeze(-1)], dim=-1)
        # np.save(f"/train33/sppro/permanent/stniu/NOTSOFAR1-Challenge-main-mask-dia/artifacts/css_mask_model30_dev2_ivecmask_split_MTG_close_talk_reb_extend5_new/{session.replace('-mc','').replace('_CH0', '')}.npy", out_mask.cpu().numpy())

        out_wav, out_mask = preds_to_wav_joint_train(mask_preds, sep_preds, window_spks, num_speaker, new_intervals, total_frame)
        
        label_init_check = label_init
        check_mask = label_init_check.get_mixture_utternce_label_informed_speaker(real_session, speaker_list, start=0, end=total_frame, max_speaker=args.max_speaker)
        # check_mask = check_mask[:, :total_frame]
        check_mask_T = check_mask.shape[-1]
        all_dur_frame = min(check_mask_T, total_frame)
        check_mask = check_mask[..., :all_dur_frame]
        out_wav = out_wav[:, :all_dur_frame, :]
        
        write_wav2 = 1
        if write_wav2:
            
            activity_th = 0.4
            # activity_dilation_sec: float = 0.4  # dilation and erosion for segmentation mask
            # activity_erosion_sec: float = 0.2
            dilation_frames = int(24)
            erosion_frames = int(12)
            activity = out_mask.mean(dim=0)  # [T, num_spks]
            # activity = torch.from_numpy(check_mask[:num_speaker,:][...].transpose(1,0)).cuda()
            # T_frame, _ = activity.shape
            # activity = torch.from_numpy(spk_mask_all[:, :T_frame].transpose(1,0)).cuda()
            
            activity_b = activity >= activity_th
            # pdb.set_trace()
            # dilate -> erode each speaker activity
            activity_final = [torch.from_numpy(erode(dilate(x.numpy(), dilation_frames), erosion_frames))
                              for x in torch.unbind(activity_b.cpu(), dim=1)]
            activity_final = torch.stack(activity_final, dim=1)[None]  # [B, T, num_spks]
            # pdb.set_trace()
            
            add_mask = 0
            if add_mask:
                out_wav = activity_final.cuda() * out_wav
            # pdb.set_trace()
            # if num_speaker > 4:
                # check_mask_rttm = '/train33/sppro/permanent/stniu/MAMSE_sep/data/MTG2_dev_mc/MULTI_MAM_SE_S2S_model.model7_30000_MTG2_dev_mc_0409_s4_fusion/rttm_th0.35_pp'
            # else:
                # check_mask_rttm = '/train33/sppro/permanent/stniu/MAMSE_sep/data/MTG2_dev_mc/MULTI_MAM_SE_S2S_model.model7_30000_MTG2_dev_mc_0409_s4_fusion/rttm_th0.50_pp'
            
            
            
            add_dia_mask = 0
            if add_dia_mask:
                check_mask_cuda = torch.from_numpy(check_mask[:num_speaker,:][None, ...].transpose(0,2,1)).cuda()
                check_mask_cuda = check_mask_cuda!= 0
                # pdb.set_trace()
                out_wav = check_mask_cuda * out_wav
            
            css_model.cpu()
            out_wav_sep = css_model.istft(out_wav.permute(2,0,1).cpu())
            css_model.cuda()
            # pdb.set_trace()
            output_path = os.path.join(args.output_dir_sep, 'css_inference', 'multichannel', session.replace('-mc','').replace('_CH0', ''))
            os.makedirs(output_path, exist_ok=True)
            # for i in range(num_speaker):
                # print(num_speaker, output_path + f'/sep_stream{str(i)}.wav')
                # write_wav(output_path + f'/sep_stream{str(i)}.wav' ,samps=out_wav_sep[i].numpy(), sr=16000)
            get_rttm(spk_num_rttm, out_wav_sep.numpy(), output_path, session.replace('_CH0', ''))
            # pdb.set_trace()
            
        # import soundfile as sf
        # pdb.set_trace()
        
        # check_point 2
        # plot_start_frame = 0
        # plot_end_frame = total_frame
        # # plot_dur_frame = 1800
        # # plot_end_frame = plot_start_frame + plot_dur_frame
        
        # plot_mask_all(check_mask[:num_speaker, plot_start_frame: plot_end_frame], session.replace('-mc','').replace('_CH0', ''), num_speaker)
        # plot_mask_all(activity[plot_start_frame: plot_end_frame,:].squeeze().permute(-1,-2).cpu().numpy(), session.replace('-mc','').replace('_CH0', '') + '_predmask', num_speaker)
        # plot_mask_all(activity_final[:, plot_start_frame: plot_end_frame,:].squeeze().permute(-1,-2).cpu().numpy(), session.replace('-mc','').replace('_CH0', '') + '_predacti', num_speaker)
        
        # pdb.set_trace()
        # preds_to_rttm(preds, new_intervals, total_frame, output_path)
    # pdb.set_trace()

def make_argparse():
    # Set up an argument parser.
    parser = argparse.ArgumentParser(description='Prepare ivector extractor weights for ivector extraction.')
    parser.add_argument('--embedding_list', metavar='PATH', required=True,
                        help='embedding_list.')  #测试集的特征列表
    parser.add_argument('--train_set_speaker_embedding_list', metavar='PATH', required=True,
                        help='train_set_speaker_embedding_list.')  #MAMSE memory中的ivector
    parser.add_argument('--feature_list', metavar='PATH', required=True,
                        help='feature_list')   #文件地址, //.../S*_U*_CH*.fea
    parser.add_argument('--model_path', metavar='PATH', required=True,
                        help='model_path.')  
    parser.add_argument('--output_dir', metavar='PATH', required=True,
                        help='output_dir.')                       
    parser.add_argument('--output_dir_sep', metavar='PATH', required=True,
                        help='output_dir.')                       
    parser.add_argument('--max_speaker', metavar='PATH', type=int, default=8,
                help='max_speaker.')
    parser.add_argument('--init_rttm', metavar='PATH', required=True,
                        help='init_rttm.')
    parser.add_argument('--model_config', metavar='PATH', type=str, default="N",
                help='domain_list.')
    parser.add_argument('--max_utt_durance', metavar='PATH', type=int, default=800*32,
                help='max_utt_durance.')
    parser.add_argument('--hop_len', metavar='PATH', type=int, default=100,
                help='hop_len.')
    parser.add_argument('--split_seg', metavar='PATH', type=int, default=0,
                help='split_seg.')
    parser.add_argument('--batch_size', metavar='PATH', type=int, default=8,
                help='batch_size.')
    parser.add_argument('--gpu', type=str, default="0",
                        help='gpu')
    parser.add_argument('--remove_overlap', action="store_true", 
                help='remove_overlap.')      
    parser.add_argument('--model_type', metavar='Model_Type', type=str, default="model_S2S",
                help='model_type.')  
    parser.add_argument('--frame_per_second', type=float, default=100, 
            help='frame_per_second.')      
    return parser


if __name__ == '__main__':
    parser = make_argparse()
    args = parser.parse_args()
    main(args)

