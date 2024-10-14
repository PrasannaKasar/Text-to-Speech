import torch as torch   # t -> torch
from utils import spectrogram2wav
from scipy.io.wavfile import write
import hyperparams as hp
from text import text_to_sequence
import numpy as np
from network import ModelPostNet, Model
from collections import OrderedDict
from tqdm import tqdm
import argparse

# Updated function to take checkpoint_path as input
def load_checkpoint(checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    new_state_dict = OrderedDict()
    
    for k, value in state_dict['model'].items():
        key = k[7:]  # Remove the "module." prefix
        new_state_dict[key] = value
        
    return new_state_dict

def synthesis(text, args):
    MODEL = Model()
    MODEL_post = ModelPostNet()
    # Load checkpoints using the paths provided via command line arguments
    MODEL.load_state_dict(load_checkpoint(args.transformer_checkpoint))
    MODEL_post.load_state_dict(load_checkpoint(args.postnet_checkpoint))
    
    text = np.asarray(text_to_sequence(text, [hp.cleaners]))
    text = torch.LongTensor(text).unsqueeze(0)
    text = text.cuda()   # pushed to gpu(cuda device)
    mel_input = torch.zeros([1, 1, 80]).cuda()
    pos_text = torch.arange(1, text.size(1) + 1).unsqueeze(0)
    pos_text = pos_text.cuda()  # pushed to gpu(cuda device)
    
    
    MODEL = MODEL.cuda()  # pushed to gpu(cuda device)
    MODEL_post = MODEL_post.cuda()  # pushed to gpu(cuda device)
    
    MODEL.eval()
    MODEL_post.eval()
    
    pbar = tqdm(range(args.max_len))  # for progress bar
    
    with torch.no_grad():
        for i in pbar:
            pos_mel = torch.arange(1, mel_input.size(1) + 1).unsqueeze(0).cuda()
            mel_pred, postnet_pred, attention, stop_token, _, attention_decoder = MODEL.forward(text, mel_input, pos_text, pos_mel)
            mel_input = torch.cat([mel_input, mel_pred[:, -1:, :]], dim=1)
            
        mag_pred = MODEL_post.forward(postnet_pred)
        
    wav = spectrogram2wav(mag_pred.squeeze(0).cpu().numpy())
    write(hp.sample_path + "/test.wav", hp.sr, wav)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--transformer_checkpoint', type=str, help='Path to the transformer checkpoint file', required=True)
    parser.add_argument('--postnet_checkpoint', type=str, help='Path to the postnet checkpoint file', required=True)
    parser.add_argument('--max_len', type=int, help='Maximum length of the generated mel-spectrogram', default=400)
    parser.add_argument('--text', type=str, help='Text input to synthesize into speech', required=True)  # New argument for text input
    
    args = parser.parse_args()
    
    # Call synthesis function with text from user input
    synthesis(args.text, args)
