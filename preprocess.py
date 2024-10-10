import hyperparams as hp
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np
from text import text_to_sequence
import collections
from scipy import signal
import torch as torch           # t -> torch
import math

class LJDatasets(Dataset):
    """ LJSpeech-1.1 dataset. """
    def __init__(self, csv_file, root_dir):
        """
        arguments:
            csv_file (string): it is the path to the csv file(name:metadata.csv) of ljspeech dataset.
            root_dir (string): it it the directory with all the raw wavs.
        
        """
        self.audio_annotations = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir
        
    def load_wav(self, filename):
        return librosa.load(filename, sr=hp.sample_rate) 
    
    def __len__(self):
        return len(self.audio_annotations)
    
    def __getitem__(self, index):
        wav_name = os.path.join(self.root_dir, self.audio_annotations.iloc[index, 0]) + '.wav'   # .iloc[row, column]
        text = self.audio_annotations.iloc[index, 1]    # .iloc[row, column]
        
        text = np.asarray(text_to_sequence(text, [hp.cleaners]), dtype=np.int32)
        melspectrogram_data = np.load(wav_name[:-4] + '.pt.npy')   # audio_01.wav -> audio_01.pt.npy
        mel_input = np.concatenate([np.zeros([1,hp.num_mels], np.float32), melspectrogram_data[:-1,:]], axis=0)
        """
            Eg: 3 time steps which is no. of rows, each time step with num_mels= 3
            mel = [
                [1,2,3]
                [3,4,5]
                [6,7,8]         
            ]
            
            mel_input = [
                [0,0,0]
                [1,2,3]
                [3,4,5]
            ]
            
        """
        text_length = len(text)
        position_of_text = np.arrange(1, text_length + 1) #position_of_text=[1,2,...,text_length(int)]
        position_of_mel = np.arrange(1, melspectrogram_data.shape[0] + 1) #position_of_mel=[1,2,...,no. of timestep] mel.shape[0]=num of time steps
        
        sample = {'text': text, 'mel': melspectrogram_data, 'text_length': text_length, 'mel_input': mel_input, 'position_of_mel': position_of_mel, 'position_of_text': position_of_text }
        
        return sample

class PostDatasets(Dataset):
    """ LJSpeech-1.1 dataset."""
    
    def __init__(self, csv_file, root_dir):
        
        """
        arguments:
            csv_file (string): it is the path to the csv file(name:metadata.csv) of ljspeech dataset.
            root_dir (string): it it the directory with all the raw wavs.
        
        """
        self.audio_annotations = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir
        
        def __len__(self):
            return len(self.audio_annotations)
        
        def __getitem__(self, index):
            wav_name = os.path.join(self.root_dir, self.audio_annotations.iloc[index, 0]) + '.wav'   # .iloc[row, column]
            melspectrogram_data = np.load(wav_name[:-4] + '.pt.npy')    # audio_01.wav -> audio_01.pt.npy
            magnitude_of_melspectrogram = np.load(wav_name[:-4] + '.mag.npy')   # audio_01.wav -> audio_01.mag.npy
            sample = {'melspectrogram_data': melspectrogram_data, 'magnitude_of_melspectrogram': magnitude_of_melspectrogram}
            
            return sample

def collate_fn_transformer(batch):

    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.Mapping):
        
        # Extracting data required from batches of dictionaries:
        text = [d['text'] for d in batch]
        melspectrogram_data = [d['melspectrogram_data'] for d in batch]
        mel_input = [d['mel_input'] for d in batch]
        text_length = [d['text_length'] for d in batch]
        position_of_mel = [d['position_of_mel'] for d in batch]
        position_of_text = [d['position_of_text'] for d in batch]
        
        # Sorting the lists in decreasing order based on text_length: 
        text = [i for i,_ in sorted(zip(text, text_length), key=lambda x: x[1], reverse=True)]
        melspectrogram_data = [i for i, _ in sorted(zip(melspectrogram_data, text_length), key=lambda x: x[1], reverse=True)]
        mel_input = [i for i, _ in sorted(zip(mel_input, text_length), key=lambda x: x[1], reverse=True)]
        position_of_text = [i for i, _ in sorted(zip(position_of_text, text_length), key=lambda x: x[1], reverse=True)]
        position_of_mel = [i for i, _ in sorted(zip(position_of_mel, text_length), key=lambda x: x[1], reverse=True)]
        text_length = sorted(text_length, reverse=True)
        
        # PAD sequence with larget length of the corresponding batch:
        text = _prepare_data(text).astype(np.int32)
        melspectrogram_data = _pad_mel(melspectrogram_data)
        mel_input = _prepare_data(mel_input)
        position_of_mel = _prepare_data(position_of_mel).astype(np.int32)
        position_of_text = _prepare_data(position_of_text).astype(np.int32)
        
        
        return torch.LongTensor(text), torch.FloatTensor(melspectrogram_data), torch.FloatTensor(mel_input), torch.LongTensor(position_of_text), torch.LongTensor(position_of_mel), torch.LongTensor(text_length)
    
    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))  #raises an error with the wrong type shown/printed

def collate_fn_postnet(batch):
    
    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.Mapping):
        
        # Extracting data required from batches of dictionaries:
        melspectrogram_data = [d['melspectrogram_data'] for d in batch]
        magnitude_of_melspectrogram = [d['magnitude_of_melspectrogram'] for d in batch]
        
        # PAD sequence with larget length of the corresponding batch:
        melspectrogram_data = _pad_mel(melspectrogram_data)
        magnitude_of_melspectrogram = _pad_mel(magnitude_of_melspectrogram)
        
        return torch.FloatTensor(melspectrogram_data), torch.FloatTensor(magnitude_of_melspectrogram)
    
    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))  #raises an error with the wrong type shown/printed
        
def _pad_data(x, length):
    _pad = 0
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

    # x = [1,2,3]
    # padded_x = _pad_data(x,5)
    # padded_x = [1,2,3,0,0]
    
def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])

    # inputs = [np.array([1, 2, 3]), np.array([4, 5]), np.array([6, 7, 8, 9])]
    # output: 
    #    [[1,2,3,0]
    #     [4,5,0,0]
    #     [6,7,8,9]]
    
def _pad_per_steps(inputs):
    timesteps = inputs.shape[-1]
    return np.pad(inputs, [[0,0],[0,0],[0, hp.outputs_per_step - (timesteps % hp.outputs_per_step)]], mode='constant', constant_values=0.0)

def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp*=x
        params+=tmp
    
    return params

def get_dataset():
    return LJDatasets(os.path.join(hp.data_path, 'metadata.csv'), os.path.join(hp.data_path,'wavs'))

def get_post_dataset():
    return PostDatasets(os.path.join(hp.data_path,'metadata.csv'), os.path.join(hp.data_path,'wavs'))

def _pad_mel(inputs):
    _pad = 0
    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        return np.pad(x, [[0, max_len - mel_len],[0,0]], mode='constant', constant_values=_pad)
    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])
