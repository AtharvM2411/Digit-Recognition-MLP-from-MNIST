"""
Digit Recognition using MLP (from scratch)
Author: Atharv
Created: 2025
# Copyright (c) 2025 Atharv
# Licensed under GAPL 3.0
This implementation is written fully in NumPy for learning purposes.
"""
#run once at start of epoch
import numpy as np
train_images=np.load('train_images.npy')#load data
train_labels=np.load('train_labels.npy')#load labels
#------------------
batch_size = 32#set as per convenience, try to keep this as a factor of total number of data
#------------------
def batching(train_images,train_labels,batch_size):#defining function of batching
    indices=np.arange(np.shape(train_images)[0])#making array of indices for each sample in dataset
    np.random.shuffle(indices)#shuffling the prev made array
    max_i=np.shape(train_images)[0]//batch_size
    for i in range(0,max_i,batch_size):#looping over indices with step=batch_size
        t=indices[i:i+batch_size]#slicing indices array 
        #taking indices[i : i + batch_size] gives you the next batch_size random samples.
        # Example: if indices = [5, 12, 3, 9, 0, 20, ...] and batch_size = 2,
        # then indices[0:2] → [5, 12], indices[2:4] → [3, 9], etc.

        train_batch_of_random_images=train_images[t]#1 batch images 
        train_batch_of_random_labels=train_labels[t]#1 batch labels

        yield train_batch_of_random_images,train_batch_of_random_labels

batches = list(batching(train_images, train_labels, batch_size))

# extract only images into numpy array
# 'batch[0]' is the image batch inside the tuple (images, labels).
# We collect them into a list, and then convert that list into a NumPy array.
# Final shape will be: (num_batches, batch_size, 784)
train_batched_random_images = np.array([batch[0] for batch in batches])

# extract only labels into numpy array
# 'batch[1]' is the label batch inside the tuple.
# Shape will become: (num_batches, batch_size, 10) for one-hot labels,
# or (num_batches, batch_size) for integer labels.
train_batched_random_labels = np.array([batch[1] for batch in batches])

np.save('train_batched_random_images.npy',train_batched_random_images)
np.save('train_batched_random_labels.npy',train_batched_random_labels)
