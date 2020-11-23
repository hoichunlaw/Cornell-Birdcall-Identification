import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
import efficientnet.keras as efn 
import librosa
import librosa.display as display
import os
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from sklearn.utils import class_weight
import warnings
from tqdm import tqdm
#from tensorflow.keras import backend

from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
from kapre.augmentation import AdditiveNoise
from kapre.time_frequency import Spectrogram

from python_speech_features import mfcc
from mutagen.mp3 import MP3
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

augmenter = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.10, p=0.5)
])

#%matplotlib inline

#!rm -r train_data
#!rm -r val_data
#!rm -r models
#!mkdir models

# suppress warnings
warnings.filterwarnings("ignore")

SOUND_DIR = "data/birdsong-recognition/train_audio/"

# function to plot signal
def plot_signal(input_path, fileName, target_audio_length=5):
    
    signal, sr = librosa.load(os.path.join(input_path, fileName), duration=12, sr=16000)
    
    signal = filter_signal(signal, sr, target_audio_length)
    
    plt.plot(list(signal))
    plt.show()
    
    print(np.max(signal))
    
def filter_signal(signal, sr, target_audio_length):

    signal_max = np.max(signal)
    frame_max = list(signal).index(signal_max)
    
    if len(signal) <= sr * target_audio_length:
        return signal
    elif frame_max + sr * target_audio_length / 2 > len(signal):
        return signal[-sr*target_audio_length:]
    elif frame_max - sr * target_audio_length / 2 < 0:
        return signal[:sr*target_audio_length]
    else:
        return signal[int(frame_max-sr*target_audio_length/2):
                      int(frame_max+sr*target_audio_length/2)]
    
def filter_signal2(signal):
    
    if signal.shape[1] <= 224:
        return signal
    
    max_energy = np.sum(signal[0:224,0:300])
    signal_max = signal[0:224,0:300]
    for i in range(signal.shape[1]-300):
        tmp = np.sum(signal[0:224, i:i+300])
        if tmp > max_energy:
            max_energy = tmp
            signal_max = signal[0:224, i:i+300]
            
    return signal_max
        
# function for creating Mel Spectrogram
def createMelSpectrogram(input_path, fileName, output_path, saveOrShow=0):
    
    # load sound signal
    signal, sr = librosa.load(os.path.join(input_path, fileName), duration=10, sr=16000)
    
    #signal = filter_signal(signal, sr, target_audio_length)
    
    # create Mel Spectrogram
    S = Melspectrogram(n_dft=1024, 
                       n_hop=320,
                       #n_hop=256,
                       input_shape=(1, signal.shape[0]),
                       padding='same', sr=sr, n_mels=224, fmin=1400, fmax=sr/2,
                       power_melgram=2.0, return_decibel_melgram=True,
                       trainable_fb=False, trainable_kernel=False)(signal.reshape(1, 1, -1)).numpy()
    
    S = S.reshape(S.shape[1], S.shape[2])
    
    print(S.shape)
    
    if saveOrShow == 0:   
        matplotlib.image.imsave(os.path.join(output_path, fileName.split(".")[0] + ".png"), S, cmap='inferno')
    else:
        #plt.imshow(S)
        #plt.show()
        display.specshow(S, sr=sr)
        plt.show()
        
def createMelSpectrogramNew(input_path, fileName, output_path_train, output_path_val):
    
    # load sound signal
    signal, sr = librosa.load(os.path.join(input_path, fileName), sr=16000, mono=True)
    
    abs_signal = [np.abs(s) for s in signal]
    rolling_5s_abs_signal = [sum(abs_signal[i*5*sr:(i*5+5)*sr]) for i in range(int(len(abs_signal) // sr // 5))]
    
    if len(signal) <= sr * 5:
        
        # add 0 padding
        signal = list(signal) + [0 for i in range(sr*5 - len(signal))]
        signal = np.array(signal, dtype=np.float32)
        
        # draw random number
        rand = np.random.randint(0, 10)
        if rand <= 7:
            output_path = output_path_train
            toAug = 1
        else:
            output_path = output_path_val
            toAug = 0
        
        if toAug == 0:
            # normal mel spectrogram
            S = Melspectrogram(n_dft=1024, n_hop=320, input_shape=(1, signal.shape[0]),
                           padding='same', sr=sr, n_mels=224, fmin=1400, fmax=sr/2,
                           power_melgram=2.0, return_decibel_melgram=True,
                           trainable_fb=False, trainable_kernel=False)(signal.reshape(1, 1, -1)).numpy()
        
            S = S.reshape(S.shape[1], S.shape[2])
            
            matplotlib.image.imsave(os.path.join(output_path, fileName.split(".")[0] + ".png"), S, cmap='inferno')
        
        else:
            # augmentation
            mySignal = augmenter(signal, sr)
            S = Melspectrogram(n_dft=1024, n_hop=320, input_shape=(1, signal.shape[0]),
                               padding='same', sr=sr, n_mels=224, fmin=1400, fmax=sr/2,
                               power_melgram=2.0, return_decibel_melgram=True,
                               trainable_fb=False, trainable_kernel=False)(mySignal.reshape(1, 1, -1)).numpy()
            S = S.reshape(S.shape[1], S.shape[2])
            
            matplotlib.image.imsave(os.path.join(output_path, fileName.split(".")[0] + "_noise.png"), 
                                    S, cmap='inferno')
        
    else:
        
        q_signal = np.quantile(rolling_5s_abs_signal, 0.75)
        
        count = 0
        
        numSamples = int((len(signal) // sr) // 5)
        
        for i in range(numSamples):
            tmpSignal = signal[int(i*5)*sr:int((i*5+5))*sr]
            
            # cut out region is highest intensity
            #window = 224 * 256
            #interval = int((sr * 5 - window) // 10)
            #intensity = [sum(tmpSignal[i*interval:i*interval+window]) for i in range(10)]
            #idx = intensity.index(max(intensity))
            #tmpSignal = tmpSignal[idx*interval:idx*interval+window]
            
            #mask = [1 if np.abs(s) > median_signal else 0 for s in tmpSignal]
                
            if sum([abs(j) for j in tmpSignal]) >= q_signal:
                
                # draw random number
                rand = np.random.randint(0, 10)
                if rand <= 7:
                    output_path = output_path_train
                    toAug = 1
                else:
                    output_path = output_path_val
                    toAug = 0
                    
                if toAug == 0:
                    S = Melspectrogram(n_dft=1024, n_hop=320, input_shape=(1, signal.shape[0]),
                                   padding='same', sr=sr, n_mels=224, fmin=1400, fmax=sr/2,
                                   power_melgram=2.0, return_decibel_melgram=True,
                                   trainable_fb=False, 
                                   trainable_kernel=False)(tmpSignal.reshape(1, 1, -1)).numpy()
            
                    S = S.reshape(S.shape[1], S.shape[2])
                    matplotlib.image.imsave(os.path.join(output_path, 
                                                     fileName.split(".")[0] + "_" + str(count) + ".png"), 
                                        S, cmap='inferno')
                else:
                    # augmentation
                    mySignal = augmenter(tmpSignal, sr)
                    S = Melspectrogram(n_dft=1024, n_hop=320, input_shape=(1, signal.shape[0]),
                                   padding='same', sr=sr, n_mels=224, fmin=1400, fmax=sr/2,
                                   power_melgram=2.0, return_decibel_melgram=True,
                                   trainable_fb=False, 
                                   trainable_kernel=False)(mySignal.reshape(1, 1, -1)).numpy()
                
                    S = S.reshape(S.shape[1], S.shape[2])
            
                    matplotlib.image.imsave(os.path.join(output_path, 
                                             fileName.split(".")[0] + "_" + str(count) + "_noise.png"), 
                                        S, cmap='inferno')
            
            count += 1
            
def addNoise(signal, noise_factor=0.50):
    noise = np.random.randn(len(signal))
    augmented_signal = signal + noise_factor * noise
    
    # Cast back to same data type
    augmented_signal = augmented_signal.astype(type(signal[0]))
    
    return augmented_signal

def aug(signal, sr):
    return augmenter(samples=signal, sample_rate=sr)

BIRDS = os.listdir("data/birdsong-recognition/train_audio/")
BIRDS = [b for b in BIRDS if b[0] != "."]

train_folder = "MelSpectrogram/train_data"
val_folder = "MelSpectrogram/val_data"

BIRDS = BIRDS[220:264]
print(len(BIRDS))

### This takes long time to run ###

# create train and val spectrogram
np.random.seed(1234)
for bird in tqdm(BIRDS):
    INPUT_DIR = os.path.join("data/birdsong-recognition/train_audio/", bird)
    TRAIN_DIR = os.path.join(train_folder, bird)
    VAL_DIR = os.path.join(val_folder, bird)
    
    # create folders
    if not(os.path.exists(TRAIN_DIR)) and not(os.path.exists(VAL_DIR)): 
        
        os.mkdir(TRAIN_DIR)
        os.mkdir(VAL_DIR)

        # split into train and val set
        for f in os.listdir(INPUT_DIR):
            #rand = np.random.randint(0, 10)
            if f[0] != ".":
                #if rand <= 7: 
                #    createMelSpectrogramNew(INPUT_DIR, f, TRAIN_DIR, toAug=1)
                #else:
                #    createMelSpectrogramNew(INPUT_DIR, f, VAL_DIR)
                createMelSpectrogramNew(INPUT_DIR, f, TRAIN_DIR, VAL_DIR)

