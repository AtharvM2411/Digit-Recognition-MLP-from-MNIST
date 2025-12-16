#!py -3.13
"""
Digit Recognition using MLP (from scratch)
Author: Atharv
Created: 2025
# Copyright (c) 2025 Atharv
# Licensed under GAPL 3.0
This implementation is written fully in NumPy for learning purposes.
"""
#Run only once brfore creating new Dataset
import numpy as np
import os
import cv2 
path = r""#paste path to data "folder"
size = (28,28)
num_of_classes = 10 #set num of classes
def preprocessing_png_to_np_array_wth_on_hot_encd(path,size,num_of_classes):
    files = sorted([f for f in os.listdir(path) if f.endswith('.png')])
    try:
        train_images=np.reshape((np.array([cv2.resize(cv2.imread(os.path.join(path,fname),cv2.IMREAD_GRAYSCALE),size) for fname in files])/255),(len(files),size[0]*size[1]))
        train_labels_unencd=np.array([int(fname[6]) for fname in files])
        train_labels_unencd = train_labels_unencd.astype(int)
        train_labels=np.zeros((len(train_images),num_of_classes),dtype=np.float32)
        train_labels[np.arange(len(train_images)),train_labels_unencd]=1

    except:
        print("Error while parsing Files.")
    return train_images,train_labels
train_images,train_labels=preprocessing_png_to_np_array_wth_on_hot_encd(path,size,num_of_classes)
np.save('train_images.npy',train_images)
np.save('train_labels.npy',train_labels)