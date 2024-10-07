import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from utils import get_spectrograms
import hyperparams as hp
import librosa

class PrepareDataset(Dataset):
    """LJSpeech dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.

        """
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0]) + '.wav'
        mel, mag = get_spectrograms(wav_name)
        
        # Change output path to use hp.output_path_used_for_prepare_data
        np.save(os.path.join(hp.output_path_used_for_prepare_data, os.path.basename(wav_name[:-4] + '.pt')), mel)
        np.save(os.path.join(hp.output_path_used_for_prepare_data, os.path.basename(wav_name[:-4] + '.mag')), mag)

        sample = {'mel':mel, 'mag': mag}

        return sample
    
if __name__ == '__main__':
    dataset = PrepareDataset(os.path.join(hp.data_path_used_for_prepare_data,'metadata.csv'), os.path.join(hp.data_path_used_for_prepare_data,'wavs'))
    dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=4)
    from tqdm import tqdm
    pbar = tqdm(dataloader)
    for d in pbar:
        pass
