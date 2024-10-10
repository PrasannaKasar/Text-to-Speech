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

def load_checkpoint(step, model_name="transformer"):
    state_dict = torch.load('./checkpoint/checkpoint_%s_%d.pth.tar'% (model_name, step))
    new_state_dict = OrderedDict()
    
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value
        
    return new_state_dict

def synthesis(text, args):
    MODEL = Model()
    MODEL_post = ModelPostNet
    
    MODEL.load_state_dict(load_checkpoint(args.restore_step1, "transformer"))
    MODEL_post.load_state_dict(load_checkpoint(args.restore_step2, "postnet"))
    
    text = np.asarray(text_to_sequence(text, [hp.cleaners]))
    text = torch.LongTensor(text).unsqueeze(0)
    text = text.cuda()   # pushed to gpu(cuda device)
    mel_input = torch.zeros([1, 1, 80]).cuda()
    pos_text = torch.arange(1, text.size(1)+1).unsqueeze(0)
    pos_text = pos_text.cuda()  #pushed to gpu(cuda device)
    
    MODEL = MODEL.cuda()  #pushed to gpu(cuda device)
    MODEL_post = MODEL_post.cuda()  #pushed to gpu(cuda device)
    
    pbar = tqdm(range(args.max_len))  # for progress bar
    
    with torch.no_grad():
        for i in pbar:
            pos_mel = torch.arange(1, mel_input.size(1)+1).unsqueeze(0).cuda()
            mel_pred, postnet_pred, attention, stop_token, _, attention_decoder = MODEL.forward(text, mel_input, pos_text, pos_mel)
            mel_input = torch.cat([mel_input, mel_pred[:,-1:,:]], dim=1)
            
        mag_pred = MODEL_post.forward(postnet_pred)
        
    wav = spectrogram2wav(mag_pred.squeeze(0).cpu().numpy())
    write(hp.sample_path + "/test.wav", hp.sr, wav)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step1', type=int, help='Global step to restore checkpoint', default=172000)
    parser.add_argument('--restore_step2', type=int, help='Global step to restore checkpoint', default=100000)
    parser.add_argument('--max_len', type=int, help='Global step to restore checkpoint', default=400)
    

    args = parser.parse_args()
    synthesis("Transformer model is so fast!",args)