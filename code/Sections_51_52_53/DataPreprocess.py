# -*- coding: utf-8 -*-
"""
Created on Tue May 16 05:58:18 2017

@author: agoswami
"""
import os
from scipy.misc import imread, imsave, imresize
from os import listdir
from os.path import isfile, join
import random
import shutil
import matplotlib.pyplot as plt
import uuid as uuid

images_raw_path = "images_raw/"
images_cooked_train_path    = "images_cooked_train/"
images_cooked_val_path      = "images_cooked_val/"
images_cooked_test_path     = "images_cooked_test/"

#delete cooked directories if they exist
if os.path.exists(images_cooked_train_path):
    shutil.rmtree(images_cooked_train_path)

if os.path.exists(images_cooked_val_path):
    shutil.rmtree(images_cooked_val_path)

if os.path.exists(images_cooked_test_path):
    shutil.rmtree(images_cooked_test_path)

images_cooked_train_metadata_filename   = "images_cooked_train_metadata.txt" 
images_cooked_val_metadata_filename     = "images_cooked_val_metadata.txt" 
images_cooked_test_metadata_filename    = "images_cooked_test_metadata.txt" 

labelsDict = {
    "burger" : 0,
    "fries" : 1,
    "burrito" : 2,
    "lasagna" : 3,
    "pizza" : 4,
    "pasta" : 5,
    "biryani" : 6,
    "sushi" : 7,
    "ramen" : 8,
    "naan" : 9,
    "sandwich" : 10,
    "dumplings" : 11,
    "icecream" : 12,
    "roastturkey" : 13,
    "dal" : 14,
    "bratwurst": 15,
    "padthai" : 16,
    "friedrice" : 17,
    "samosa" : 18,
    "cordonbleu" : 19,
}

#labelsDict = {
#    "cats" : 0,
#    "dogs" : 1,
#    "giraffe" : 2,
#}

with open(images_cooked_train_metadata_filename, "w") as f_train, \
   open(images_cooked_val_metadata_filename, "w") as f_val, \
   open(images_cooked_test_metadata_filename, "w") as f_test:
    
    for label in labelsDict:
        skipped_exceptionerror = 0
        skipped_resizeerror = 0
        skipped_not = 0
        
        images_raw_path_withlabel = images_raw_path + label
        
        if(os.path.isdir(images_raw_path_withlabel) == False):
            continue
        
        filenames = [f for f in listdir(images_raw_path_withlabel) if isfile(join(images_raw_path_withlabel, f))]
#        print(filenames)
        
        for fname in filenames:
            file_raw = images_raw_path_withlabel + "/" + fname

#with gzip.open(file, 'rb') as in_file:
#    s = in_file.read()
#
## Now store the uncompressed data
#path_to_store = file[:-3]  # remove the '.gz' from the filename
#
## store uncompressed file data from 's' variable
#with open(path_to_store, 'wb') as f:
#    f.write(s)

#            img = imread(file_raw)
            try:
                img = imread(file_raw)
                imgresize = imresize(img, (32, 32))
            except:
#                print("continuing..1")
                skipped_exceptionerror += 1
                continue
            
            if (imgresize.shape != (32,32,3)):
#                print("continuing..2")
                skipped_resizeerror += 1
                continue
            
            skipped_not += 1
            _r = random.random()
#            print("{0}:{1}".format(label, skipped_not))

            if (_r < 0.7):
#               save resized image in 'images_cooked_train' folder, and make entry in metadata file
                file_new = images_cooked_train_path + label + '/' + str(uuid.uuid4()) + ".jpg"
                file_new_dir = os.path.dirname(file_new)
                if not os.path.exists(file_new_dir):
                    os.makedirs(file_new_dir)
                    
                imsave(file_new, imgresize)
                f_train.write(str(labelsDict[label]) + "," + label + "," + file_new + "\n")
                
            elif (_r > 0.7 and _r < 0.9):    
                file_new = images_cooked_val_path + label + '/' + str(uuid.uuid4()) + ".jpg"
                file_new_dir = os.path.dirname(file_new)
                if not os.path.exists(file_new_dir):
                    os.makedirs(file_new_dir)
                    
                imsave(file_new, imgresize)
                f_val.write(str(labelsDict[label]) + "," + label + "," + file_new + "\n")
                
            else:
                file_new = images_cooked_test_path + label + '/' + str(uuid.uuid4()) + ".jpg"
                file_new_dir = os.path.dirname(file_new)
                if not os.path.exists(file_new_dir):
                    os.makedirs(file_new_dir)
        
                imsave(file_new, imgresize)
                f_test.write(str(labelsDict[label]) + "," + label + "," + file_new + "\n")

#            print("{0}:{1}".format(file_raw, file_new))

        print("{0}:{1}:{2}:{3}".format(label, skipped_exceptionerror, skipped_resizeerror, skipped_not))