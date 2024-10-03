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
    wav_name = os.path.join(self.root_dir, self.audio_annotations.ix[index, 0]) + '.wav'   # .ix[row, column]
    text = self.audio_annotations.ix[index, 1]    # .ix[row, column]
    
    text = np.asarray(text_to_sequence(text, [hp.cleaners]), dtype=np.int32)
    mel = np.load(wav_name[:-4] + '.pt.npy')   # audio_01.wav -> audio_01.pt.npy
    mel_input = np.concatenate([np.zeros([1,hp.num_mels], np.float32), mel[:-1,:]], axis=0)
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
    position_of_mel = np.arrange(1, mel.shape[0] + 1) #position_of_mel=[1,2,...,no. of timestep] mel.shape[0]=num of time steps
    
    sample = {'text': text, 'mel': mel, 'text_length': text_length, 'mel_input': mel_input, 'position_of_mel': position_of_mel, 'position_of_text': position_of_text }
    
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
        
        
        

