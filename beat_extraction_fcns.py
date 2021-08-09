import numpy as np
import pandas as pd
import pickle
import glob
from scipy.signal import find_peaks
import os
import multiprocessing as mp
from scipy.signal import savgol_filter

def normalize(signal):
    new_signal = (signal - np.min(signal))/(np.max(signal) - np.min(signal))
    return new_signal

def detect_peaks(signal):
    yhat = savgol_filter(signal, 21, 2) # window size 51, polynomial order 3
    sm_new = signal - yhat
    peaks, properties = find_peaks(np.abs(sm_new), distance=15, prominence=0.32)
    beat_sigs = []
    for peak in peaks:
        left_edge =peak - 19
        right_edge = peak + 19
        if left_edge >= 0 and right_edge <= (len(signal) - 1):
            beat_cut = signal[left_edge:right_edge]
            beat_sigs.append(normalize(beat_cut))
    return peaks, beat_sigs


def extract_beats_from_many(signals):
    pool = mp.Pool(processes=os.cpu_count())
    results = [pool.apply_async(detect_peaks, args=(row,)) for row in signals]
    output = [p.get() for p in results]
    pool.close()
    all_beats = []
    for pair in output:
        all_beats.extend(pair[1])
    all_beats = np.array(all_beats)
    return all_beats