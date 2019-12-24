# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 02:29:44 2019
File: data_downloading.py
Author: Travis Tang (Voon Hao)
Github: https://github.com/travistangvh
Description: Downloading the FER2013 dataset
Note: This code is heavily adapted from Amine Horseman's the 'Convert FER2013 to Image and Landmarks' in the Facial Expression Recognition SVM git. 
(From: https://github.com/amineHorseman/facial-expression-recognition-svm/blob/master/convert_fer2013_to_images_and_landmarks.py)
"""

import numpy as np
import pandas as pd
import os
import errno
import imageio

# initialization
image_height = 48
image_width = 48
window_size = 24
window_step = 6
SAVE_IMAGES = True
SELECTED_LABELS = [0,1,2,3,4,5,6]
IMAGES_PER_LABEL = 500
OUTPUT_FOLDER_NAME = "../datasets/"

def data_download():
    # loading Dlib predictor and preparing arrays:
    print( "preparing")
    #predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    original_labels = [0, 1, 2, 3, 4, 5, 6]
    new_labels = list(set(original_labels) & set(SELECTED_LABELS))
    nb_images_per_label = list(np.zeros(len(new_labels), 'uint8'))
    try:
        os.makedirs(OUTPUT_FOLDER_NAME)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(OUTPUT_FOLDER_NAME):
            pass
        else:
            raise
            
    
    print( "importing csv file")
    
    data = pd.read_csv('../datasets/raw/fer2013.csv')
    
    for category in data['Usage'].unique():
        print( "converting set: " + category + "...")
        # create folder
        if not os.path.exists(category):
            try:
                os.makedirs(OUTPUT_FOLDER_NAME + '/' + category)
            except OSError as e:
                if e.errno == errno.EEXIST and os.path.isdir(OUTPUT_FOLDER_NAME):
                   pass
                else:
                    raise
        
        # get samples and labels of the actual category
        category_data = data[data['Usage'] == category]
        samples = category_data['pixels'].values
        labels = category_data['emotion'].values
        
        # get images and extract features
        images = []
        labels_list = []
        landmarks = []
        hog_features = []
        hog_images = []
        for i in range(len(samples)):
            try:
                if labels[i] in SELECTED_LABELS: 
                    image = np.fromstring(samples[i], dtype=int, sep=" ").reshape((image_height, image_width))
                    images.append(image)
                    imageio.imwrite(OUTPUT_FOLDER_NAME + '/' + category + '/' + str(i) + '.jpg', image)
                    
            except Exception as e:
                print( "error in image: " + str(i) + " - " + str(e))
    
        np.save(OUTPUT_FOLDER_NAME + '/' + category + '/images.npy', images)
