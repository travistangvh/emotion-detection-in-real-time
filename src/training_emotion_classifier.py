# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 02:29:44 2019
File: training_emotion_classifier.py
Author: Travis Tang (Voon Hao)
Github: https://github.com/travistangvh
Description: Training of CNN model for emotion classification
Note: The following code is adapted from Machine Learning Mastery in the link https://machinelearningmastery.com/check-point-deep-learning-models-keras/
"""

import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from keras_vggface.vggface import VGGFace
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from src import label_and_dir
from src import data_generator
from src import data_downloading

#downloading data from Kaggle
data_downloading()

#importing data
train_dir, valid_dir, test_dir, train_label, valid_label, test_label = label_and_dir()
train_generator, valid_generator, test_generator = data_generator(train_dir, valid_dir, test_dir, train_label, valid_label, test_label)

#custom parameters
nb_class = 7
hidden_dim = 1024

#Creating a VGGFace model instance.
vgg_model = VGGFace(include_top=False, input_shape=(96, 96, 3))

#Use the architecture of the VGGFace and append a fully connected layer with 1024 neurons before the final classification using softmax.
last_layer = vgg_model.get_layer('pool5').output
x = layers.Flatten()(last_layer)
x = layers.Dense(hidden_dim, activation='relu', name='fc7')(x)
x = layers.Dense(nb_class, activation='softmax', name='fc8')(x)
custom_vgg_model = Model(vgg_model.input, x)

#Printing the model summary
custom_vgg_model.summary()

# Training the model with fer2013 data.
custom_vgg_model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.0001),
              metrics=['accuracy'])

#Creating a callback that saves a model when the validation loss decreases from the previous epoch.
filepath="../trained_models/weights-improvement-{epoch:02d}-{val_loss:.3f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only = False)
callbacks = [checkpoint]

#training the model
history = custom_vgg_model.fit_generator(
            train_generator,
            validation_data = valid_generator,
            steps_per_epoch = 897,
            epochs = 100,
            validation_steps = 897,
            verbose = 2,
            callbacks=callbacks)

