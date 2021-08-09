import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA
from beat_extraction_fcns import *

class The_Autoencoder:

    def __init__(self, chunks_list=None, group_list=None, encoded_dim=20,
                encoder_filename=None):
        self._encoded_dim = encoded_dim
        if chunks_list is not None:
            self._chunk_arr = np.array(chunks_list)
            self._all_groups = np.array(group_list)
            self.break_data()
            self.build_autoencoder()
            self.train_autoencoder()
        if encoder_filename is not None:
            with open(encoder_filename, 'rb') as handle:
                self._encoder = pickle.load(handle)
            handle.close()
        
        
        
        
    def expand_chunks(self, chunk_arr):
        break_list = []
        for chunk in chunk_arr:
            for i in range(chunk.shape[1]):
                this_piece = chunk[:,i]
                this_piece = (this_piece - np.min(this_piece))/((np.max(this_piece) - np.min(this_piece)) + 0.000001)
                break_list.append(this_piece)
        break_list = np.array(break_list)
        return break_list
    
    
    
    def break_data(self):
        self._X_train = self.expand_chunks(self._chunk_arr)
        self._X_train = extract_beats_from_many(self._X_train)
        
        
    def build_autoencoder(self):
        self._encoder = IncrementalPCA(n_components=self._encoded_dim, 
                                       whiten=True)
        
    
    def train_autoencoder(self):
        self._encoder = self._encoder.fit(self._X_train)
        
        
    
    def encode(self, chunk_arr, num_beats_to_encode=10):
        chunk_encode_rows = []
        for chunk in chunk_arr:
            for col in range(chunk.shape[1]):
                peaks, beat_sigs = detect_peaks(chunk[:,col])
                if len(beat_sigs) > 0:
                    beat_pca = self._encoder.transform(np.array(beat_sigs))
                flat_sig = np.zeros(num_beats_to_encode*self._encoded_dim)
                idx = 0
                count = 0
                if len(beat_sigs) > 0:
                    for row in beat_pca:
                        flat_sig[idx:idx+beat_pca.shape[1]] = row
                        idx += beat_pca.shape[1]
                        count += 1 
                        if count == num_beats_to_encode:
                            break
                if col == 0:
                    flat_chunk = flat_sig
                else:
                    flat_chunk = np.concatenate((flat_chunk, flat_sig))
            chunk_encode_rows.append(flat_chunk)
        chunk_encode_rows = np.array(chunk_encode_rows)
        return chunk_encode_rows
                      
                      
    def save(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self._encoder, handle)
        handle.close()
                      
    