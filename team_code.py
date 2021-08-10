#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of the required functions, remove non-required functions, and add your own functions.

################################################################################
#
# Imported functions and variables
#
################################################################################

# Import functions. These functions are not required. You can change or remove them.
from helper_code import *
import numpy as np, os, sys, joblib
import pandas as pd
from slice_dice import Slice_Dice
from Train_Autoencoder import The_Autoencoder
from Encode_Shape import Encode_Shape
from sklearn.neighbors import KNeighborsClassifier
from OVR_DNN import OVR_DNN
import pickle

# Define the Challenge lead sets. These variables are not required. You can change or remove them.
twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
four_leads = ('I', 'II', 'III', 'V2')
three_leads = ('I', 'II', 'V2')
two_leads = ('I', 'II')
lead_sets = (twelve_leads, six_leads, four_leads, three_leads, two_leads)

################################################################################
#
# Training model function
#
################################################################################

# Train your model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    # Find header and recording files.
    print('Finding header and recording files...')

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    if not num_recordings:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    # Extract features and labels from dataset.
    print('Extracting features and labels...')
    
    # data extraction parameters
    desire_rate = 50 #Hz, desired sampling rate
    time_window = 5 #sec, window of time of ekg to look in
    overlap = 0.25 # proportion of overlap between time windows

    # loop through all recordings
    for i in range(num_recordings):
        print('    {}/{}...'.format(i+1, num_recordings))

        # Load header and recording.
        this_header = load_header(header_files[i])
        recording = load_recording(recording_files[i])
        this_recording = recording.T # transpose data time is rows and columns are features

        # get labels and frequency
        this_freq = get_frequency(this_header)
        this_labels = get_labels(this_header)
        
        # shape data and labels for single recording
        print('  slicing and dicing data...')
        slicer = Slice_Dice(this_recording, this_freq)
        chunks = slicer.process_chunks(desire_rate, 
                                       time_window, 
                                       overlap, 
                                       this_header)
        chunk_labels = [this_labels]*len(chunks)
        group = [recording_files[i]]*len(chunks)
        print('     found', len(chunks), '5 second windows in this file...')
        
        # combine all recordings
        if i == 0:
            all_chunks = chunks
            all_labels = chunk_labels
            all_groups = group
        else:
            all_chunks.extend(chunks)
            all_labels.extend(chunk_labels)
            all_groups.extend(group)

    print('Training the encoder model...')
    my_encoder = The_Autoencoder(all_chunks, all_groups)

    # Train a model for each lead set.
    for leads in lead_sets:
        print('Training model for {}-lead set: {}...'.format(len(leads), ', '.join(leads)))

        shaper = Encode_Shape(my_encoder, all_chunks, leads, 
                          chunk_labels=all_labels)
        features, labels = shaper.get_shaped_output()
        print('  found features of shape:', features.shape)
        print('  found labels of shape:', labels.shape)

        # Train the model.
        my_ovr = OVR_DNN(X_train=features, y_train=labels)

        # Save the model.
        save_model(model_directory, leads, shaper, my_ovr)

################################################################################
#
# Running trained model function
#
################################################################################

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def run_model(model, header, recording):
    leads = model['leads']
    classifier = model['classifier']
    my_encoder = model['encoder']
    
    # Load features
    desire_rate = 50 #Hz, desired sampling rate
    time_window = 5 #sec, window of time of ekg to look in
    overlap = 0.95 # proportion of overlap between time windows
    this_recording = recording.T
    this_freq = get_frequency(header)
    slicer = Slice_Dice(this_recording, this_freq)
    chunks = slicer.process_chunks(desire_rate, 
                                       time_window, 
                                       overlap, 
                                       header)
    shaper = Encode_Shape(my_encoder, chunks, leads, test=True)
    features = shaper.get_shaped_output()

    # Predict labels and probabilities
    many_predicts = classifier.predict(features)
    probabilities = np.mean(many_predicts, axis=0)
    print(probabilities)
    
    labels = np.zeros(len(probabilities)).astype(np.int)
    labels[probabilities >= 0.5] = 1
    print(labels)
    
    classes = shaper._scored_labels
    print(classes)
    return classes, labels, probabilities

################################################################################
#
# File I/O functions
#
################################################################################

# Save a trained model. This function is not required. You can change or remove it.
def save_model(model_directory, leads, shaper_object, my_ovr):
    filename = get_model_filename(leads)
    filename = os.path.join(model_directory, filename)
    # save encoder
    filename_tf = filename.split('.pick')[0] + '_encoder.pickle'
    shaper_object._encoder.save(filename_tf)
    # save ovr model files
    my_ovr.save_models(filename)
    # save leads
    model_dict = {
        'leads': leads,
    }
    with open(filename, 'wb') as handle:
        pickle.dump(model_dict, handle)
    handle.close()
    

# Load a trained model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def load_model(model_directory, leads):
    filename = get_model_filename(leads)
    filename = os.path.join(model_directory, filename)
    # load encoder
    filename_tf = filename.split('.pick')[0] + '_encoder.pickle'
    encoder_object = The_Autoencoder(encoder_filename=filename_tf)
    # load ovr models
    my_ovr = OVR_DNN(filename=filename)
    # load leads
    with open(filename, 'rb') as handle:
        saved_dict = pickle.load(handle)
    handle.close()
    model_dict = {
        'encoder': encoder_object,
        'leads': saved_dict['leads'],
        'classifier': my_ovr
    }
    return model_dict

# Define the filename(s) for the trained models. This function is not required. You can change or remove it.
def get_model_filename(leads):
    sorted_leads = sort_leads(leads)
    return 'model_' + '-'.join(sorted_leads) + '.pickle'
