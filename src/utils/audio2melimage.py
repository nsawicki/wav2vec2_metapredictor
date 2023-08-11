import numpy as np 
import pandas as pd
import os
import torch
import librosa
from librosa import display
from librosa.feature import mfcc

def audio2MelImage(audio,sampling_rate,num_frequency_bins,nfcc_num,time_samp):

    S = librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_mels=num_frequency_bins)
    Mel = librosa.power_to_db(S, ref=np.max)/10+4

    output_image = torch.zeros((nfcc_num,time_samp)).type(torch.FloatTensor)
    Ssum = np.sum(Mel,axis=0)
    MaxE = np.argmax(Ssum)
    if MaxE > Mel.shape[1]-64 : 
        MaxE = Mel.shape[1]-65
    if MaxE< 64 :
        MaxE = 64
    if Mel.shape[1] > time_samp:
        output_image = torch.from_numpy(Mel[:,MaxE-64:MaxE+64])
    else: 
        output_image[:,:Mel.shape[1]] = torch.from_numpy(Mel)

    output_image = output_image.double()
    return output_image
