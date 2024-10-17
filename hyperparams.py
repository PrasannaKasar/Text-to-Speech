# Audio
num_mels = 80
# num_freq = 1024
n_fft = 2048
sr = 22050
# frame_length_ms = 50.
# frame_shift_ms = 12.5
preemphasis = 0.97
frame_shift = 0.0125 # seconds
frame_length = 0.05 # seconds
hop_length = int(sr*frame_shift) # samples.
win_length = int(sr*frame_length) # samples.
n_mels = 80 # Number of Mel banks to generate
power = 1.2 # Exponent for amplifying the predicted magnitude
min_level_db = -100
ref_level_db = 20
hidden_size = 256
embedding_size = 256
max_db = 100
ref_db = 20
    
n_iter = 60
# power = 1.5
outputs_per_step = 1

epochs = 350
lr = 0.001
save_step = 5000
image_step = 500
batch_size = 64

cleaners='english_cleaners'

# General:
data_path = '/kaggle/input/mel-mag-prepared-data'
checkpoint_path = '/kaggle/working'
sample_path = '/kaggle/working'

# For kaggle use:
"""
    data_path_used_for_prepare_data = '/kaggle/input/the-lj-speech-dataset/LJSpeech-1.1'
    output_path_used_for_prepare_data = '/kaggle/working/'
    
"""

# For local use: (prepare_data.ipynb)
data_path_used_for_prepare_data = 'C:\\Users\\admin\\Desktop\\LJSpeech-1.1'
output_path_used_for_prepare_data = 'C:\\Users\\admin\\Desktop\\LJSpeech-1.1'