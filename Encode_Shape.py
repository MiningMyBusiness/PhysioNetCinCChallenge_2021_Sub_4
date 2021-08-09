import numpy as np
import pandas as pd


class Encode_Shape:
    
    def __init__(self, encoder_object, chunk_arrs, leads, chunk_labels=None, test=False):
        self._twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
        self._six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
        self._four_leads = ('I', 'II', 'III', 'V2')
        self._three_leads = ('I', 'II', 'V2')
        self._two_leads = ('I', 'II')
        self._leads = leads
        self._encoder = encoder_object
        self._chunk_arrs = chunk_arrs
        self._chunk_labels = chunk_labels
        self.create_label_map()
        if not test:
            self.remove_leads()
        
        
    def remove_leads(self):
        last_idx = self._chunk_arrs[0].shape[1] - 1
        sec_last_idx = last_idx - 1
        feature_indices = np.array([self._twelve_leads.index(lead) for lead in self._leads] + [sec_last_idx, 
                                                                                               last_idx])
        new_chunk_arrs = [None]*len(self._chunk_arrs)
        for i,chunk in enumerate(self._chunk_arrs):
            new_chunk_arrs[i] = chunk[:, feature_indices]
        self._chunk_arrs = new_chunk_arrs
        print(len(self._chunk_arrs))
        print(self._chunk_arrs[0].shape)
        if self._chunk_labels is not None:
            print(len(self._chunk_labels))
            
            
    def get_shaped_output(self):
        self._chunk_encode_row = self._encoder.encode(self._chunk_arrs)
        if self._chunk_labels is None:
            return self._chunk_encode_row
        else:
            self.assign_label()
            return self._chunk_encode_row, self._label_one_hot
        
    
    def create_label_map(self):
        scored_df = pd.read_csv('dx_mapping_scored.csv')
        scored_labels = list(scored_df['SNOMEDCTCode'])
        label_to_idx = {}
        idx_to_label = {}
        count = 0
        for label in scored_labels:
            label_to_idx[label] = count
            idx_to_label[count] = label
            count += 1
        self._scored_labels = scored_labels
        self._label_to_idx = label_to_idx
        self._idx_to_label = idx_to_label
        
        
    def assign_label(self):
        self._chunk_labels = np.array(self._chunk_labels)
        in_scored_arr = np.zeros(len(self._chunk_arrs)).astype(bool)
        for i,labels in enumerate(self._chunk_labels):
            in_scored = False
            for label in labels:
                if int(label) in self._scored_labels:
                    in_scored = True
            in_scored_arr[i] = in_scored
        print('  found', np.sum(in_scored_arr), 'data points with scores.')
        self._chunk_encode_row = self._chunk_encode_row[in_scored_arr]
        self._chunk_labels = self._chunk_labels[in_scored_arr]
        self.one_hot_encode_label()
        
        
        
    def one_hot_encode_label(self):
        label_one_hot = np.zeros((len(self._chunk_labels), len(self._scored_labels)))
        for i,labels in enumerate(self._chunk_labels):
            for label in labels:
                if int(label) in self._scored_labels:
                    label_one_hot[i,self._label_to_idx[int(label)]] = 1.0
        self._label_one_hot = label_one_hot
        