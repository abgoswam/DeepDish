# -*- coding: utf-8 -*-
"""
Created on Tue May 16 18:51:37 2017

@author: agoswami
"""

import os
from scipy.misc import imread, imsave, imresize
from os import listdir
from os.path import isfile, join
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt

images_cooked_train_metadata_filename   = "images_cooked_train_metadata.txt" 
images_cooked_val_metadata_filename     = "images_cooked_val_metadata.txt" 
images_cooked_test_metadata_filename    = "images_cooked_test_metadata.txt" 

H = 32
W = 32
C = 3

def loadData():
    X_train, y_train, P_train    = loadDataFromMetadata(images_cooked_train_metadata_filename)
    X_val, y_val, P_val          = loadDataFromMetadata(images_cooked_val_metadata_filename)
    X_test, y_test, P_test       = loadDataFromMetadata(images_cooked_test_metadata_filename)
   
    return X_train, y_train, P_train, X_val, y_val, P_val, X_test, y_test, P_test
    
def loadDataFromMetadata(metadata_filename):
    with open(metadata_filename, "r") as f:
        N = sum(1 for line in f)
    
    X = np.zeros((N, H, W, C))
    y = np.zeros(N, dtype=np.uint8)
    P = []
    
    with open(metadata_filename, "r") as f:
        line = f.readline().strip()
#        print(line)
        i = 0
        while(line):
            label, label_name, path = line.split(',')
#            print("{0}:{1}".format(label, path))
            
            img = imread(path)
            if (img.shape != (32,32,3)):
                line = f.readline().strip()
                continue
        
#            plt.imshow(img)
#            plt.show()
            X[i] = img.astype(np.float64)
            y[i] = int(label)
            P.append(path)
            line = f.readline().strip()
            i += 1
#            break
            
    return X, y, P

############ DEBUG #########
#X_list = []
#y_list = []
#with open(images_cooked_train_metadata_filename, "r") as f:
#    line = f.readline().strip()
#    
#    while(line):
#        label, path = line.split(',')
#        print("{0}:{1}".format(label, path))
#        
#        img = imread(path)
#        if (img.shape != (32,32,3)):
#            line = f.readline().strip()
#            continue
#        
#        X_list.append(img)
#        y_list.append(label)
#        line = f.readline().strip()
#
#print("abhishek")
#
#n = len(X_list)
#h, w, c = X_list[0].shape
#X = np.zeros((n, h, w, c))
#
#for i in range(n):
#    print(i)
#    X[i] = X_list[i]
#    
#y = np.array(y_list)
###############


if __name__ == "__main__":

    X_train, y_train, X_val, y_val, X_test, y_test = loadData()
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)