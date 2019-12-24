# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 02:29:44 2019
File: data_generator.py
Author: Travis Tang (Voon Hao)
Github: https://github.com/travistangvh
Description: Preprocessing data (into dataframe) for the training of the keras model
Note: A tutorial on the flow_from_dataframe method can be found at https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def data_generator(train_dir, valid_dir, test_dir, train_label, valid_label, test_label):
    # Add our data-augmentation parameters to ImageDataGenerator
    train_datagen = ImageDataGenerator(#rescale = 1./255.,
                                       rotation_range = 40,
                                       width_shift_range = 0.2,
                                       height_shift_range = 0.2,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)
    
    # Note that the validation data should not be augmented!
    valid_datagen = ImageDataGenerator()#rescale = 1./255. )
    test_datagen = ImageDataGenerator()#rescale = 1./255. )
    
    # Flow training images in batches of 5 using train_datagen generator
    train_generator=train_datagen.flow_from_dataframe(dataframe=train_label,
                                                     directory=train_dir,
                                                     x_col="id",
                                                     y_col="emotion",
                                                     target_size=(96,96),
                                                     batch_size=32,
                                                     seed=42,
                                                     shuffle=True,
                                                     class_mode="categorical",
                                                     color_mode='rgb')
    
    # # Flow validation images in batches of 5 using test_datagen generator
    valid_generator=valid_datagen.flow_from_dataframe(dataframe=valid_label,
                                                      directory=valid_dir,
                                                      x_col="id",
                                                      y_col="emotion",
                                                      target_size=(96,96),
                                                      batch_size=32,
                                                      seed=42,
                                                      shuffle=True,
                                                      class_mode="categorical",
                                                      color_mode='rgb')
    
    # Flow validation images in batches of 5 using test_datagen generator
    test_generator=test_datagen.flow_from_dataframe(dataframe=test_label,
                                                     directory=test_dir,
                                                     x_col="id",
                                                     y_col=None,
                                                     target_size=(96,96),
                                                     batch_size=32,
                                                     seed=42,
                                                     shuffle=False,
                                                     class_mode=None,
                                                     color_mode='rgb')
return train_generator, valid_generator, test_generator
