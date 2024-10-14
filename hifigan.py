import torch
from utils import spectrogram2wav
from scipy.io.wavfile import write
import hyperparams as hp
from text import text_to_sequence
import numpy as np
from network import ModelPostNet, Model
from collections import OrderedDict
from tqdm import tqdm
import argparse
from speechbrain.inference.vocoders import HIFIGAN
import torchaudio

# Updated function to take checkpoint_path as input
def load_checkpoint(checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    new_state_dict = OrderedDict()

    for k, value in state_dict['model'].items():
        key = k[7:]  # Remove the "module." prefix
        new_state_dict[key] = value

    return new_state_dict

def synthesis(text, args):
    MODEL = Model().cuda()
    MODEL_post = ModelPostNet().cuda()

    # Load checkpoints using the paths provided via command line arguments
    MODEL.load_state_dict(load_checkpoint(args.transformer_checkpoint))
    MODEL_post.load_state_dict(load_checkpoint(args.postnet_checkpoint))

    # Prepare input text
    text = np.asarray(text_to_sequence(text, [hp.cleaners]))
    text = torch.LongTensor(text).unsqueeze(0).cuda()  # Push to GPU

    mel_input = torch.zeros([1, 1, 80]).cuda()
    pos_text = torch.arange(1, text.size(1) + 1).unsqueeze(0).cuda()  # Push to GPU

    MODEL.eval()
    MODEL_post.eval()

    pbar = tqdm(range(args.max_len))  # For progress bar

    with torch.no_grad():
        for i in pbar:
            pos_mel = torch.arange(1, mel_input.size(1) + 1).unsqueeze(0).cuda()
            mel_pred, postnet_pred, _, _, _, _ = MODEL.forward(text, mel_input, pos_text, pos_mel)
            mel_input = torch.cat([mel_input, mel_pred[:, -1:, :]], dim=1)

        mag_pred = MODEL_post.forward(postnet_pred)

    # Generate waveform using HiFi-GAN
    wav = generate_audio_with_hifigan(mag_pred)

    # Write output WAV file
    write(hp.sample_path + "/test.wav", hp.sr, wav)

def generate_audio_with_hifigan(mag_pred):
    # Load pre-trained HiFi-GAN model from SpeechBrain
    hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="pretrained_models/tts-hifigan-ljspeech")

    # Convert magnitude spectrogram to the required input format for HiFi-GAN
    mel = torch.tensor(mag_pred).unsqueeze(0).float().cuda()  # Add batch dimension and push to GPU

    # Generate audio
    with torch.no_grad():
        generated_audio = hifi_gan.decode_batch(mel).squeeze(1).cpu().numpy()  # Convert to numpy array

    return generated_audio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transformer_checkpoint', type=str, help='Path to the transformer checkpoint file', required=True)
    parser.add_argument('--postnet_checkpoint', type=str, help='Path to the postnet checkpoint file', required=True)
    parser.add_argument('--max_len', type=int, help='Maximum length of the generated mel-spectrogram', default=400)
    parser.add_argument('--text', type=str, help='Text input to synthesize into speech', default="Hello! Good Morning")

    args = parser.parse_args()

    # Call synthesis function with text from user input
    synthesis(args.text, args)
