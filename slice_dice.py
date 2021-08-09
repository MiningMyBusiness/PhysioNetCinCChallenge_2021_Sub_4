from helper_code import *
import numpy as np
import pandas as pd
import pickle 
from scipy.signal import resample


class Slice_Dice:
    
    def __init__(self, signal, samp_rate):
        self._signal = signal
        self._samp_rate = samp_rate
    

    def scale_signal(self):
        self._signal = (self._signal - np.min(self._signal))/(np.max(self._signal) - np.min(self._signal))

    
    def resample_signal(self, desire_rate):
        desire_num_samps = (desire_rate/self._samp_rate)*len(self._signal)
        self._signal = resample(self._signal, int(desire_num_samps))
        self._samp_rate = desire_rate
    
    
    def dice(self, time_window, overlap):
        window = int(time_window*self._samp_rate)
        if overlap < 1.0:
            step_size = int((1 - overlap)*window)
        else:
            step_size = 1
        chunks = []
        if len(self._signal) > window:
            for i in range(0, len(self._signal) - window, step_size):
                this_chunk = self._signal[i:i+window]
                chunks.append(this_chunk)
        else:
            new_signal = np.zeros((window, self._signal.shape[1]))
            new_signal[:len(self._signal)] = self._signal
            chunks = [new_signal]
        return chunks
    
    
    def process_chunks(self, desire_rate, time_window, overlap, header):
        self.resample_signal(desire_rate)
        self.scale_signal()
        self._feat_build = Feature_Build(self._signal, header)
        self._signal = self._feat_build.add_feats()
        self._signal[np.isnan(self._signal)] = 0.0
        self._signal[self._signal < 0] = 0.0
        self._signal[self._signal > 1.0] = 0.0
        chunks = self.dice(time_window, overlap)
        for i,chunk in enumerate(chunks):
            chunk = chunk[1:249,:]
            chunks[i] = chunk
        return chunks
    
    
    

    
class Feature_Build:
    
    def __init__(self, signal, header):
        self._signal = signal
        self._header = header
        self._features = None
        
        
    def add_age(self):
        age = get_age(self._header)
        if age is None:
            age = 0
        age = age/200
        age_arr = np.ones((len(self._signal), 1))*age
        if self._features is None:
            self._features = np.concatenate((self._signal,
                                        age_arr), axis=1)
        else:
            self._features = np.concatenate((self._features,
                                        age_arr), axis=1)
        
        
    def add_sex(self):
        sex = get_sex(self._header)
        if sex in ('Female', 'female', 'F', 'f'):
            sex = 0
        elif sex in ('Male', 'male', 'M', 'm'):
            sex = 1
        else:
            sex = 0.5
        sex_arr = np.ones((len(self._signal), 1))*sex
        if self._features is None:
            self._features = np.concatenate((self._signal,
                                        sex_arr), axis=1)
        else:
            self._features = np.concatenate((self._features,
                                        sex_arr), axis=1)
        
    
    def add_feats(self):
        self.add_age()
        self.add_sex()
        return self._features
        
