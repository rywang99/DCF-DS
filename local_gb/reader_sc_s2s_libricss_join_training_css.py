# -*- coding: utf-8 -*-

import torch
import numpy as np
import os
import math
import scipy.io as sio
import json
import copy
import HTK
import re
import tqdm
import pdb
import librosa
from css_conformer.training.augmentations import MicShiftAugmentation

def plot_wav(data_wav, target_speaker_wav, label_data):
    # ch = 1
    import matplotlib.pyplot as plt
    import soundfile as sf
    sf.write(f'./sample/input.wav', data_wav[1].numpy().T, 16000)
    for i in range(3):
        # pdb.set_trace()
        bb = label_data[1][i][np.newaxis,:].numpy()
        plot_label = bb.repeat(513, axis = 0)
        # plot_speaker_mask = speaker_mask[i].T
        plt.imshow( plot_label, cmap='binary' , interpolation = 'none')
        if i == 0:
            plt.colorbar()
        # plt.colorbar()
        plt.savefig(f'./sample/mask_label{i}.png')
        #pdb.set_trace()
        sf.write(f'./sample/target_out{str(i)}.wav', target_speaker_wav[1][i].numpy().T, 16000)
        
        # plt.imshow( plot_speaker_mask, cmap='binary' , interpolation = 'none')
        # plt.savefig(f'mask_speaker{i}.png')



class LoadIVector():
    def __init__(self, speaker_embedding_txt):
        self.speaker_embedding = self.load_ivector(speaker_embedding_txt)

    def load_ivector(self, speaker_embedding_txt):
        SCP_IO = open(speaker_embedding_txt)
        speaker_embedding = {}
        raw_lines = [l for l in SCP_IO]
        SCP_IO.close()
        for i in tqdm.tqdm(range(len(raw_lines) // 2), ncols=100):
            speaker = raw_lines[2*i].split()[0]
            ivector = np.array(raw_lines[2*i+1].split()[:-1], np.float32)
            speaker_embedding[speaker] = torch.from_numpy(ivector)
        return speaker_embedding

    def get_speaker_embedding(self, speaker):
        if not speaker in self.speaker_embedding.keys():
            print("{} not in sepaker embedding list".format(speaker))
            # pdb.set_trace()
        return self.speaker_embedding[speaker]

def collate_fn_mask(batch):
    '''
    batch: B * (data, embedding, label)
    data: [T, F]
    speaker_embedding: [num_speaker, embedding_dim]
    mask_label: [num_speaker, T, C]
    '''
    # data, speaker_embedding, mask_label, wav_data, target_speaker_wav
    batch = [ b for b in batch if b != None ]
    num_speaker = batch[0][1].shape[0]
    length = [item[2].shape[1] for item in batch]
    ordered_index = sorted(range(len(length)), key=lambda k: length[k], reverse = True) #B个, 按照长度排序
    #print(ordered_index)
    nframes = []
    input_data = []
    speaker_embedding = []
    label_data = []
    speaker_index = np.array(range(num_speaker))
    #print("speaker_index :", speaker_index)
    #print("ordered_index :", ordered_index)

    Time, Freq = batch[ordered_index[0]][0].shape  #用最长的开矩阵
    batch_size = len(length)
    input_data = np.zeros([batch_size, Time, Freq]).astype(np.float32)
    mask_data = np.zeros([batch_size, num_speaker, Time]).astype(np.float32)
    
    
    Time = batch[ordered_index[0]][3].shape[-1]
    num_channel = batch[ordered_index[0]][3].shape[-2]
    # batch_size = len(length)
    # pdb.set_trace()
    data_wav = np.zeros([batch_size, num_channel, Time]).astype(np.float32)
    target_speaker_wav = []
    
    for i, id in enumerate(ordered_index):
        np.random.shuffle(speaker_index)
        #print("shuffle speaker_index", speaker_index)
        input_data[i, :length[id], :] = batch[id][0]
        # pdb.set_trace()
        data_wav[i, :batch[id][3].shape[-2], :batch[id][3].shape[-1]] = batch[id][3]
        
        
        speaker_embedding.append(batch[id][1][speaker_index])
        mask = copy.deepcopy(batch[id][2][speaker_index]) # (N, T)
        overlap = np.sum(mask>0, axis=0)
        mask[:, overlap>1] = 0    #0和大于1的都mask
        mask_data[i, :, :length[id]] = mask
        #print(batch[id][2].shape)
        label_data.append(torch.from_numpy(batch[id][2][speaker_index].astype(np.float32)))
        nframes.append(length[id])
        # pdb.set_trace()
        target_speaker_wav.append(torch.from_numpy(batch[id][4][speaker_index].astype(np.float32)))

    # pdb.set_trace()
    # pdb.set_trace()


    input_data = torch.from_numpy(input_data).transpose(1, 2) # B * T * F => B * F * T
    data_wav = torch.from_numpy(data_wav) # (B, ch, t)
    speaker_embedding = torch.stack(speaker_embedding) # B * Speaker * Embedding_dim
    mask_data = torch.from_numpy(mask_data) # B * nspeaker * T
    label_data = torch.stack(label_data)  # B * nspeaker * T
    # pdb.set_trace()
    target_speaker_wav = torch.stack(target_speaker_wav) # (B, nspeaker, ch, t)
    # print(torch.sum(label_data))
    # plot_wav(data_wav, target_speaker_wav, label_data)
    # pdb.set_trace()
    MicShift = 1
    if MicShift:
        augmentation_fns = MicShiftAugmentation(seed=59438191 + 2112)
        # pdb.set_trace()
        data_wav = augmentation_fns(data_wav.permute(0,2,1)).permute(0,2,1) # B, T, C [Batch, T, Mics]
        target_speaker_wav = augmentation_fns(target_speaker_wav.permute(0,3,2,1)).permute(0,3,2,1) # B, T, C, SPK [Batch, T, Mics, Spks]
        # pdb.set_trace()
        
    # ref_mic = 0
    # target_speaker_wav = target_speaker_wav[:,:,ref_mic,:]
    
    # pdb.set_trace()
    return input_data, speaker_embedding, mask_data, label_data, nframes, data_wav, target_speaker_wav

class Fbank_Embedding_Label_Mask():
    def __init__(self, feature_scp, speaker_embedding_txt, label=None, train_segments=None, embedding_type='ivector', append_speaker=False, diff_speaker=False, min_speaker=0, max_speaker=2, max_utt_durance=800, frame_shift=None, min_segments=0, mixup_rate=0, alpha=0.5, frame_per_second=62.5):
        self.feature_scp = feature_scp
        self.max_utt_durance = max_utt_durance
        if frame_shift == None:
            self.frame_shift = self.max_utt_durance // 2
        else:
            self.frame_shift = frame_shift
        self.label = label
        self.append_speaker = append_speaker
        self.min_speaker = min_speaker
        self.max_speaker = max_speaker
        self.diff_speaker = diff_speaker
        self.mixup_rate = mixup_rate #mixup_rate<0 means not perform mixup strategy when training
        self.alpha = alpha
        self.frame_per_second = frame_per_second
        self.feature_list, self.session_to_feapath, self.session_to_wavpath = self.get_feature_info(train_segments, min_segments)
        # pdb.set_trace()
        self.session_to_feature_list = self.session_to_feature(self.feature_list)
        # pdb.set_trace()
        self.speaker_embedding = LoadIVector(speaker_embedding_txt)
        self.data_set_speaker = list(self.speaker_embedding.speaker_embedding.keys())


    def prepare_segment(self, session, cur_frame, cur_end):
        feature_list = []
        while(cur_frame < cur_end):
            if cur_frame + self.max_utt_durance < cur_end:
                feature_list.append((session, cur_frame, cur_frame+self.max_utt_durance))
                cur_frame += self.frame_shift
            else:
                cur_frame = max(0, cur_end-self.max_utt_durance)
                feature_list.append((session, cur_frame, cur_end))
                break
        return feature_list

    def get_feature_info(self, train_segments=None, min_segments=0, random_start=False):
        feature_list = []
        session_to_feapath = {}
        session_to_wavpath = {}
        segments = {}
        if train_segments != None:
            with open(train_segments) as IN:
                for l in IN:
                    session, s ,e = l.rstrip().split()
                    if session not in segments.keys():
                        segments[session] = []
                    segments[session].append([int(s), int(e)])
        with open(self.feature_scp) as SCP_IO:
            files = [ l for l in SCP_IO ]
            for l in tqdm.tqdm(files, ncols=100):
                '''
                basename: 0000.fea
                '''
                session = os.path.basename(l).rstrip().replace('.fea', '')
                session_to_feapath[session] = l.rstrip()
                real_session = session
                # pdb.set_trace()
                wavpath = f'/train33/sppro/permanent/stniu/data/wav/css-datasets/train/mixture/{session}.wav'
                session_to_wavpath[session] = wavpath
                # real_session = re.sub("_CH.*", "", real_session)
                # if(self.label != None and self.label.mixture_num_speaker(session) < self.min_speaker):
                #     continue
                # pdb.set_trace()
                try:
                    total_frame = HTK.readHtk_info(l.rstrip())[0]
                    if total_frame < self.max_utt_durance:
                        continue
                except:
                    print('line 143', l)
                    continue
                if self.label != None:
                    wav_dur = librosa.get_duration(filename=wavpath)
                    total_frame2 = int((wav_dur * 16000 - 80) / 16000 * self.frame_per_second)
                    # session_length = self.label.get_session_length(session)
                    session_length = self.label.get_session_length(real_session)
                    MAX_LEN = min(total_frame, total_frame2, session_length)
                    # MAX_LEN = min(total_frame, session_length)
                    if MAX_LEN < self.max_utt_durance:
                        continue
                else:
                    # MAX_LEN = total_frame
                    print('line 158 no label')
                    continue
                s = [0, MAX_LEN]
                if random_start:
                    cur_frame = np.random.randint(s[0], s[0]+self.max_utt_durance-self.frame_shift)
                else:
                    cur_frame = s[0]
                cur_end = min(s[1], MAX_LEN)
                if cur_end - cur_frame < min(self.max_utt_durance, min_segments): continue
                feature_list += self.prepare_segment(session, cur_frame, cur_end)
                
        return feature_list, session_to_feapath, session_to_wavpath

    def session_to_feature(self, feature_list):
        session_to_feature_list = {}
        for l in feature_list:
            session = l[0]
            if session not in session_to_feature_list.keys():
                session_to_feature_list[session] = []
            session_to_feature_list[session].append(l)
        return session_to_feature_list

    def load_fea(self, path, start, end):
        try:
            # pdb.set_trace()
            nSamples, sampPeriod, sampSize, parmKind, data = HTK.readHtk_start_end(path, start, end)
        except:
            print("line175 {} {} {}".format(path, start, end))

        htkdata= np.array(data).reshape(end - start, int(sampSize / 4))
        return end - start, htkdata

    def load_wav_fea(self, path, start, end, sr=16000, norm=True):
        start = start * int(sr / self.frame_per_second)
        end = end * int(sr / self.frame_per_second) + 80
        
        wav = HTK.read_wav_start_end(path, start, end)
        #print(path, start, end, len(wav))
        return end - start, wav.astype(np.float32)

    def __len__(self):
        return len(self.feature_list)
        
    def load_all_channel(self, wav_path, start, end):
        channel_num = 7
        wav_data_list = []
        for i in range(channel_num):
            cur_wav = wav_path.replace('CH0', f'CH{str(i)}')
            _, wav_data = self.load_wav_fea(cur_wav, start, end)
            wav_data_list.append(wav_data)
            # print(cur_wav)
            # pdb.set_trace()
        return np.stack(wav_data_list, axis=0)
        # pdb.set_trace()
    
    def __getitem__(self, idx):
        channel_num = 7
        l = self.feature_list[idx]
        session, start, end = l
        # session, start, end = 'ES2012a.Array1-08', 59400, 60200
        real_session = session
        # real_session = re.sub("_CH.*", "", real_session)
        path = self.session_to_feapath[session]
        wavpath = self.session_to_wavpath[session]
        # load feature (T * F)
        # pdb.set_trace()
        # wav_data = self.load_all_channel(wavpath, start, end)
        
        _, data = self.load_fea(path, start, end)
        wav_data = self.load_all_channel(wavpath, start, end) # (ch, T)
        # _, wav_data = self.load_wav_fea(wavpath, start, end)
        # load label (Speaker * T * 3)
        # pdb.set_trace()
        try:
            mask_label, speakers = self.label.get_mixture_utternce_label(real_session, self.speaker_embedding, start=start, end=end)
            mask_list = []
            for speaker in speakers:
                # pdb.set_trace()
                wav_id = '.'.join(session.split('.')[:-1])
                target_wav_path = f'/train33/sppro/permanent/stniu/data/wav/css-datasets/train/gt_spk_direct_early_echoes/{wav_id}.gt_spk_direct_early_echoes_SPK{speaker}_CH0.wav'
                target_wav_data = self.load_all_channel(target_wav_path, start, end) # (ch, T)
                # _, target_wav_data = self.load_wav_fea(target_wav_path, start, end)
                mask_list.append(target_wav_data)
            # pdb.set_trace()
            target_speaker_wav = np.stack(mask_list, axis=0) # (spk, ch, T)
        except:
                # pdb.set_trace()
            print('line196', l)
            return None
        # pdb.set_trace()
        # load embedding (Speaker * Embedding_dim)
        # pdb.set_trace()
        if np.random.uniform() <= self.mixup_rate:
            #print(len(self.session_to_feature_list[session]))
            #pdb.set_trace()
            session, start, end = self.session_to_feature_list[session][np.random.choice(range(len(self.session_to_feature_list[session])))]
            _, data_2 = self.load_fea(path, start, end)
            wav_data_2 = self.load_all_channel(wavpath, start, end) # (ch, T)
            # _, wav_data_2 = self.load_wav_fea(wavpath, start, end)
            try:
                mask_label_2, speakers_2 = self.label.get_mixture_utternce_label(real_session, self.speaker_embedding, start=start, end=end)
                mask_list2 = []
                for speaker in speakers_2:
                    # target_wav_path = f'/train33/sppro/permanent/stniu/data/wav/css-datasets/train/gt_spk_direct_early_echoes/{session}_gt_spk_direct_early_echoes_SPK{speaker}_CH0.wav.wav'
                    wav_id = '.'.join(session.split('.')[:-1])
                    target_wav_path = f'/train33/sppro/permanent/stniu/data/wav/css-datasets/train/gt_spk_direct_early_echoes/{wav_id}.gt_spk_direct_early_echoes_SPK{speaker}_CH0.wav'
                    target_wav_data = self.load_all_channel(target_wav_path, start, end) # (ch, T)
                    # _, target_wav_data = self.load_wav_fea(target_wav_path, start, end)
                    mask_list2.append(target_wav_data)
                target_speaker_wav2 = np.stack(mask_list2, axis=0)
            except:
                print('line 208', l)
                return None
            if speakers != speakers_2:
                print("not in a same session")
                return None
            # pdb.set_trace()
            weight = np.random.beta(self.alpha, self.alpha)
            data = weight * data + (1 - weight) * data_2
            wav_data = weight * wav_data + (1 - weight) * wav_data_2
            mask_label = weight * mask_label + (1 - weight) * mask_label_2
            target_speaker_wav = weight * target_speaker_wav + (1 - weight) * target_speaker_wav2
        # pdb.set_trace()
        speaker_embedding = []
        for speaker in speakers:
            speaker_embedding.append(self.speaker_embedding.get_speaker_embedding("{}-{}".format(session, speaker)))
        # pdb.set_trace()
        num_speaker, T = mask_label.shape
        wav_T = wav_data.shape[1]
        #pdb.set_trace()
        #print(mask_label.shape)
        data_set_speaker1 = list(self.speaker_embedding.speaker_embedding.keys())
        if self.append_speaker and (num_speaker < self.max_speaker):
            append_label = np.zeros([self.max_speaker - num_speaker, T])
            #append_label[:, :, 0] = 1
            mask_label = np.vstack([mask_label, append_label])
            append_target_wav = np.zeros([self.max_speaker - num_speaker, channel_num, wav_T])
            # pdb.set_trace()
            target_speaker_wav = np.vstack([target_speaker_wav, append_target_wav])
            for speaker in speakers:
                try:
                    data_set_speaker1.remove("{}-{}".format(session, speaker))
                    # print('{}-{}'.format(session, speaker))
                except:
                    print('line 234: ', speaker)
            for speaker in np.random.choice(data_set_speaker1, self.max_speaker - num_speaker, replace=False):
                speaker_embedding.append(self.speaker_embedding.get_speaker_embedding(speaker))
        speaker_embedding = torch.stack(speaker_embedding)
        # pdb.set_trace()
        if num_speaker > self.max_speaker:
            speaker_index = np.array(range(num_speaker))
            np.random.shuffle(speaker_index)
            speaker_embedding = speaker_embedding[speaker_index][:self.max_speaker]
            mask_label = mask_label[speaker_index][:self.max_speaker]
            target_speaker_wav = target_speaker_wav[speaker_index][:self.max_speaker]
            # pdb.set_trace()
            # wav_data = wav_data[speaker_index][:self.max_speaker]
        '''
        returns:
        data: [T, F]
        speaker_embedding: [num_speaker, embedding_dim]
        mask_label: [num_speaker, T, C]
        '''
        # pdb.set_trace()
        # if flag:
            # if flag:
        # import matplotlib.pyplot as plt
        # import soundfile as sf
        # for i in range(8):
            # plot_label = mask_label[i][np.newaxis,:].repeat(513, axis = 0)
            # # plot_speaker_mask = speaker_mask[i].T
            # plt.imshow( plot_label, cmap='binary' , interpolation = 'none')
            # if i == 0:
                # plt.colorbar()
            # # plt.colorbar()
            # plt.savefig(f'mask_label{i}.png')
            # # plt.imshow( plot_speaker_mask, cmap='binary' , interpolation = 'none')
            # # plt.savefig(f'mask_speaker{i}.png')
            # sf.write(f'./out{str(i)}.wav', target_speaker_wav[i,:], 16000)
            # # sf.write('./out2.wav', target_speaker_wav[2,:], 16000)
        # pdb.set_trace()
        # sf.write('./input.wav', wav_data, 16000)
        # sf.write('./out1.wav', target_speaker_wav[1,:], 16000)
        # sf.write('./out2.wav', target_speaker_wav[2,:], 16000)
        return data, speaker_embedding, mask_label, wav_data, target_speaker_wav

class RTTM_to_Speaker_Mask():
    def __init__(self, oracle_rttm, differ_silence_inference_speech=False, max_speaker=8, frame_per_second=50):
        self.differ_silence_inference_speech = differ_silence_inference_speech
        self.frame_label = self.get_label(oracle_rttm, frame_per_second)
        self.max_speaker = max_speaker

    def get_label(self, oracle_rttm, frame_per_second=100):
        '''
        SPEAKER session0_CH0_0L 1  116.38    3.02 <NA> <NA> 5683 <NA>
        '''
        IO = open(oracle_rttm)
        files = [ l for l in IO ]
        IO.close()
        MAX_len = {}
        rttm = {}
        utts = []
        speech = np.array([1])
        for line in tqdm.tqdm(files, ncols=100):
            line = line.split(" ")
            session = line[1]
            spk = line[7]
            if not session in MAX_len.keys():
                MAX_len[session] = 0
            start = int(float(line[3]) * frame_per_second)
            end = int(float(line[4]) * frame_per_second) + start
            if end > MAX_len[session]:
                MAX_len[session] = end
            utts.append([session, spk, start, end])
        for utt in tqdm.tqdm(utts, ncols=100):
            session, spk, start, end = utt
            if not session in rttm.keys():
                rttm[session] = {}
            if not spk in rttm[session].keys():
                rttm[session][spk] = np.zeros(MAX_len[session], dtype=np.int8)
                #rttm[session][spk][:, 0] = 1 #2 dim
            rttm[session][spk][start: end] = speech
        return rttm
    
    def get_label2(self, oracle_rttm, frame_per_second=100):
        '''
        SPEAKER session0_CH0_0L 1  116.38    3.02 <NA> <NA> 5683 <NA>
        '''
        files = open(oracle_rttm)
        MAX_len = {}
        rttm = {}
        self.all_speaker_list = []
        for line in files:
            line = line.split(" ")
            session = line[1]
            if not session in MAX_len.keys():
                MAX_len[session] = 0
            start = int(float(line[3]) * frame_per_second)
            end = int(float(line[4]) * frame_per_second) + start
            if end > MAX_len[session]:
                MAX_len[session] = end
        files.close()
        files = open(oracle_rttm)
        for line in files:
            line = line.split(" ")
            session = line[1]
            spk = line[7]
            self.all_speaker_list.append(spk)
            if not session in rttm.keys():
                rttm[session] = {}
            if not spk in rttm[session].keys():
                    rttm[session][spk] = np.zeros([MAX_len[session], 2], dtype=np.int8)
            #print(line[3])
            start = int(float(line[3]) * frame_per_second)
            end = int(float(line[4]) * frame_per_second) + start
            rttm[session][spk][start: end, 1] = 1
        for session in rttm.keys():
            for spk in rttm[session].keys():
                rttm[session][spk][:, 0] = 1 - rttm[session][spk][:, 1]
        self.all_speaker_list = list(set(self.all_speaker_list))
        files.close()
        # pdb.set_trace()
        return rttm
            
    def mixture_num_speaker(self, session):
        #print(session)
        return len(self.frame_label[session])

    def get_session_length(self, session):
        for spk in self.frame_label[session].keys():
            return len(self.frame_label[session][spk])

    def get_mixture_utternce_label(self, session, speaker_embedding, raw_session=None, start=0, end=None, check=True):
        speakers = []
        mixture_utternce_label = []
        speaker_duration = []
        speaker = []
        if raw_session == None:
            raw_session = session
        for spk in sorted(self.frame_label[session].keys()):
            speaker_duration.append(np.sum(self.frame_label[session][spk])) # speaker order 1,2,3,4
            speaker.append(spk)
        speaker_duration_id_order = sorted(list(range(len(speaker_duration))), reverse=True, key=lambda k:speaker_duration[k])
        cur_num_speaker = 0
        for spk_idx in speaker_duration_id_order:
            spk = speaker[spk_idx]
            if check:
                try:
                    if "{}-{}".format(raw_session, spk) not in speaker_embedding.speaker_embedding.keys():
                        #print("{}-{}".format(session, spk))
                        continue
                except:
                    if spk not in speaker_embedding:
                        #print("{}-{}".format(session, spk))
                        continue
            if end > len(self.frame_label[session][spk]):
                print("{}-{}: {}/{}".format(session, spk, end, len(self.frame_label[session][spk])))
            mixture_utternce_label.append(self.frame_label[session][spk][start:end])
            speakers.append(spk)
            cur_num_speaker += 1
            if cur_num_speaker >= self.max_speaker:
                break
        # pdb.set_trace()
        return np.vstack(mixture_utternce_label).reshape(len(speakers), end - start)[sorted(range(len(speakers)), key=lambda k:speakers[k])], sorted(speakers) # (n_spk, len)
    
    def get_mixture_utternce_label_informed_speaker(self, session, speaker_list, start=0, end=None, max_speaker=4):
        mixture_utternce_label = []
        for spk in speaker_list:
            try:
                mixture_utternce_label.append(self.frame_label[session][spk][start:end])
            except:
                # print('line367', session, self.frame_label.keys())
                # real_session = session.split("_")[0]
                real_session = re.sub("_CH.*", "", session)
                # real_session = session
                mixture_utternce_label.append(self.frame_label[real_session][spk][start:end])
        mask_label = np.stack(mixture_utternce_label)
        num_speaker, T = mask_label.shape
        if num_speaker < max_speaker:
            append_label = np.zeros([max_speaker - num_speaker, T])
            #append_label[:, :, 0] = 1
            mask_label = np.vstack([mask_label, append_label])
        return mask_label

    def get_mixture_utternce_label_single_speaker(self, session, target_speaker=None, start=0, end=None):
        target_speaker = target_speaker.split('_')[-1]
        if target_speaker != None:
            return self.frame_label[session][target_speaker][start: end]
            
class RTTM_to_Speaker_Mask_add_frame_per_second():
    def __init__(self, oracle_rttm, max_speaker=4, frame_per_second=50):
        self.frame_label = self.get_label(oracle_rttm, frame_per_second)
        self.max_speaker = max_speaker

    def get_label(self, oracle_rttm, frame_per_second=100):
        '''
        SPEAKER session0_CH0_0L 1  116.38    3.02 <NA> <NA> 5683 <NA>
        '''
        files = open(oracle_rttm)
        MAX_len = {}
        rttm = {}
        self.all_speaker_list = []
        for line in files:
            line = line.split(" ")
            session = line[1]
            if not session in MAX_len.keys():
                MAX_len[session] = 0
            start = int(float(line[3]) * frame_per_second)
            end = int(float(line[4]) * frame_per_second) + start
            if end > MAX_len[session]:
                MAX_len[session] = end
        files.close()
        files = open(oracle_rttm)
        for line in files:
            line = line.split(" ")
            session = line[1]
            spk = line[-3]
            self.all_speaker_list.append(spk)
            if not session in rttm.keys():
                rttm[session] = {}
            if not spk in rttm[session].keys():
                    rttm[session][spk] = np.zeros([MAX_len[session], 2], dtype=np.int8)
            #print(line[3])
            start = int(float(line[3]) * frame_per_second)
            end = int(float(line[4]) * frame_per_second) + start
            rttm[session][spk][start: end, 1] = 1
        for session in rttm.keys():
            for spk in rttm[session].keys():
                rttm[session][spk][:, 0] = 1 - rttm[session][spk][:, 1]
        self.all_speaker_list = list(set(self.all_speaker_list))
        files.close()
        # pdb.set_trace()
        return rttm
            
    def mixture_num_speaker(self, session):
        return len(self.frame_label[session])

    def get_session_length(self, session):
        for spk in self.frame_label[session].keys():
            return len(self.frame_label[session][spk])

    def get_mixture_utternce_label(self, session, speaker_embedding, start=0, end=None, check=True):
        speakers = []
        mixture_utternce_label = []
        speaker_duration = []
        speaker = []
        for spk in sorted(self.frame_label[session].keys()):
            speaker_duration.append(np.sum(self.frame_label[session][spk][:, 1])) # speaker order 1,2,3,4
            speaker.append(spk)
        speaker_duration_id_order = sorted(list(range(len(speaker_duration))), reverse=True, key=lambda k:speaker_duration[k])
        cur_num_speaker = 0
        for spk_idx in speaker_duration_id_order:
            spk = speaker[spk_idx]
            if check:
                try:
                    if "{}-{}".format(session, spk) not in speaker_embedding.speaker_embedding.keys():
                        #print("{}-{}".format(session, spk))
                        continue
                except:
                    if spk not in speaker_embedding:
                        #print("{}-{}".format(session, spk))
                        continue
            if end > len(self.frame_label[session][spk]):
                print("{}-{}: {}/{}".format(session, spk, end, len(self.frame_label[session][spk])))
            mixture_utternce_label.append(self.frame_label[session][spk][start:end, :])
            speakers.append(spk)
            cur_num_speaker += 1
            if cur_num_speaker >= self.max_speaker:
                break
        return np.vstack(mixture_utternce_label).reshape(len(speakers), end - start, -1)[sorted(range(len(speakers)), key=lambda k:speakers[k])], sorted(speakers)
    
    def get_mixture_utternce_label_informed_speaker(self, session, speaker_list, start=0, end=None, max_speaker=4):
        mixture_utternce_label = []
        for spk in speaker_list:
            try:
                mixture_utternce_label.append(self.frame_label[session][spk][start:end, :])
            except:
                real_session = session.split("_")[0]
                mixture_utternce_label.append(self.frame_label[real_session][spk][start:end, :])
        mask_label = np.stack(mixture_utternce_label)
        num_speaker, T, C = mask_label.shape
        if num_speaker < max_speaker:
            append_label = np.zeros([max_speaker - num_speaker, T, C])
            append_label[:, :, 0] = 1
            mask_label = np.vstack([mask_label, append_label])
        return mask_label

    def get_mixture_utternce_label_single_speaker(self, session, target_speaker=None, start=0, end=None):
        target_speaker = target_speaker.split('_')[-1]
        if target_speaker != None:
            return self.frame_label[session][target_speaker][start: end, :]            
