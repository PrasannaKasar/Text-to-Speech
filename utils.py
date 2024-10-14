import numpy as np
import librosa
import os, copy
from scipy import signal
import hyperparams as hp
import torch as torch    # t -> torch
import numpy as np
import librosa

def get_spectrograms(fpath):
    '''Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.
    Args:
      fpath: A string. The full path of a sound file.
      hp: An object containing hyperparameters.
    Returns:
      mel: A 2D array of shape (T, n_mels) and dtype of float32.
      mag: A 2D array of shape (T, 1+n_fft/2) and dtype of float32.
    '''
    # Load audio file
    y, sr = librosa.load(fpath, sr=hp.sr)

    # Trim leading and trailing silence
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # Compute the Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=hp.n_mels, n_fft=hp.n_fft, hop_length=hp.hop_length)

    # Convert to decibels
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Normalize Mel spectrogram
    mel_spectrogram_db = np.clip((mel_spectrogram_db - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mel_spectrogram_db = mel_spectrogram_db.T.astype(np.float32)  # Transpose to (T, n_mels)

    # Compute the STFT
    D = librosa.stft(y, n_fft=hp.n_fft, hop_length=hp.hop_length)

    # Calculate the magnitude spectrum
    magnitude_spectrum = np.abs(D)

    # Convert magnitude spectrum to dB
    magnitude_spectrum_db = 20 * np.log10(np.maximum(1e-5, magnitude_spectrum))

    # Normalize magnitude spectrum
    magnitude_spectrum_db = np.clip((magnitude_spectrum_db - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    magnitude_spectrum_db = magnitude_spectrum_db.T.astype(np.float32)  # Transpose to (T, 1+n_fft/2)
    # print(f'mel_spec shape = {mel_spectrogram_db.shape}, mag_spec shape = {magnitude_spectrum_db.shape}')
    return mel_spectrogram_db, magnitude_spectrum_db

def spectrogram2wav(mag):
    '''# Generate wave file from linear magnitude spectrogram
    Args:
      mag: A numpy array of (T, 1+n_fft//2)
    Returns:
      wav: A 1-D numpy array.
    '''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag**hp.power)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)

def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.'''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y

def invert_spectrogram(spectrogram):
    '''Applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]
    '''
    # You might need to create a phase array here if it's not part of your input
    # Assuming phase is zero initially (for simplicity), you could use a random phase.
    phase = np.exp(1j * np.zeros_like(spectrogram))  # Initialize phase to zeros
    complex_spec = spectrogram * phase  # Reconstruct the complex STFT
    return librosa.istft(complex_spec, hp.hop_length, win_length=hp.win_length, window="hann")

def get_positional_table(d_pos_vec, n_position=1024):
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def guided_attention(N, T, g=0.2):
    '''Guided attention. Refer to page 3 on the paper.'''
    W = np.zeros((N, T), dtype=np.float32)
    for n_pos in range(W.shape[0]):
        for t_pos in range(W.shape[1]):
            W[n_pos, t_pos] = 1 - np.exp(-(t_pos / float(T) - n_pos / float(N)) ** 2 / (2 * g * g))
    return W
