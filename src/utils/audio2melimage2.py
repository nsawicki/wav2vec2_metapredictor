import numpy as np 
import pandas as pd
import os
import torch
import librosa
import cv2

from librosa import display
from librosa.feature import mfcc
from matplotlib import pyplot as plt

def renderMelImage(audio_filepath,sampling_rate,num_frequency_bins,mfcc_num,time_samp):

    y, sr = librosa.load(audio_filepath)

    window_size = 1024
    window = np.hanning(window_size)
    #stft  = librosa.core.spectrum.stft(y, n_fft=window_size, hop_length=100, window=window)
    #out = 2 * np.abs(stft) / np.sum(window)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=window_size,hop_length=50,window=window)
    out = 2* np.abs(S) / np.sum(window)
    print(np.shape(out))

    # For plotting headlessly
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax, y_axis='log', x_axis='time')
    fig.savefig('../../vizualizations/mel_image.png')

def audio2MelImage(audio,sampling_rate,window_size,hop_length,num_frequency_bins,time_samps,convert_to_rgb_image=False,load_audio=False,render_png=False):

    window = np.hanning(window_size)

    if load_audio == True:
        audio,tmp_fs = librosa.load(audio)

    #S = librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_mels=num_frequency_bins)
    S = librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_fft=window_size,n_mels=window_size,hop_length=50,window=window)
    #out = librosa.power_to_db(S, ref=np.max)/10+4
    out = 2 * np.abs(S) / np.sum(window)
    if convert_to_rgb_image == False:
        out_resize = cv2.resize(out,(time_samps,num_frequency_bins))
    if convert_to_rgb_image == True:
        out_resize = cv2.resize(out,(time_samps,num_frequency_bins))
        out_resize = cv2.cvtColor(out_resize,cv2.COLOR_GRAY2RGB)

    if convert_to_rgb_image == False:
        output_image = torch.empty(num_frequency_bins,time_samps).type(torch.FloatTensor)
        output_image[:,:] = torch.from_numpy(librosa.amplitude_to_db(out_resize,ref=np.max))
    elif convert_to_rgb_image == True:
        output_image = torch.empty(num_frequency_bins,time_samps,3).type(torch.FloatTensor)
        output_image[:,:,:] = torch.from_numpy(librosa.amplitude_to_db(out_resize,ref=np.max)).squeeze()

    #final_output = cv2.resize(output_image, (num_frequency_bins,time_samps))
    final_output = output_image.double()
   
    if render_png == True:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        p = librosa.display.specshow(librosa.amplitude_to_db(final_output[:,:,1], ref=np.max), ax=ax, y_axis='log', x_axis='time')
        fig.savefig('../../vizualizations/mel_image.png')
    
    return final_output

if __name__ == '__main__':

    #renderMelImage('../../data/raw_coughs/C011-AW9-QAR-MBN-37F-cough-3.caf',16000,128,128,128)
    audio2MelImage('../../data/raw_coughs/0480-843bc1528bfa4d8a8109c315dbffaa0a700aebd81748d4641e324ffb68784f89-cough-8.caf',16000,1024,50,512,128,convert_to_rgb_image=True,load_audio=True,render_png=True)
