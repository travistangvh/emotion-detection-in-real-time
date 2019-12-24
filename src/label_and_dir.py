# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 02:10:55 2019

@author: User
"""
import pandas as pd
from src import append_ext

def label_and_dir():
    train_label = pd.read_csv('../datasets/labels/traininglabel.csv',dtype=str)
    valid_label = pd.read_csv('../datasets/labels/publictestlabel.csv',dtype=str)
    test_label = pd.read_csv('../datasets/labels/privatetestlabel.csv',dtype=str)
    
    train_label["id"]=train_label["id"].apply(append_ext)
    valid_label["id"]=valid_label["id"].apply(append_ext)
    test_label["id"]=test_label["id"].apply(append_ext)
    
    # Define our example directories and files
    train_dir = '../datasets/Training'
    valid_dir = '../datasets/PublicTest'
    test_dir = '../datasets/PrivateTest'
    return train_dir, valid_dir, test_dir, train_label, valid_label, test_label