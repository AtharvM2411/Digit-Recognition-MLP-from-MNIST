"""
Digit Recognition using MLP (from scratch)
Author: Atharv
Created: 2025
# Copyright (c) 2025 Atharv
# Licensed under GAPL 3.0
This implementation is written fully in NumPy for learning purposes.
"""
#Run only Once brfore creating new dataset
import numpy as np
path="data/mnist"#paste path of .idx
def load_idx_images(path):
    with open(path, "rb") as f:#b in rb means binary read
        #reads main structure of .idx3-ubyte file in bytes
        magic = int.from_bytes(f.read(4), "big")
        num_images = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")

        #reads main structure of .idx1-ubyte file in bytes
        data = f.read(rows * cols * num_images)
        images = np.frombuffer(data, dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
        return images#size of array (num of images , 28, 28) and values from 0 to 255

def load_idx_labels(path):
    with open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        num_labels = int.from_bytes(f.read(4), "big")
        data = f.read(num_labels)
        labels = np.frombuffer(data, dtype=np.uint8)
        return labels

#make actual np array
train_images = load_idx_images("data/mnist/train-images.idx3-ubyte")
train_labels = load_idx_labels("data/mnist/train-labels.idx1-ubyte")

test_images = load_idx_images("data/mnist/t10k-images.idx3-ubyte")
test_labels = load_idx_labels("data/mnist/t10k-labels.idx1-ubyte")

# Normalize: [0..255] → [0..1]
train_images = train_images.astype(np.float32) / 255.0 #conversion of 255 values to 0-1
test_images  = test_images.astype(np.float32) / 255.0 #conversion of 255 values to 0-1

# Flatten: (N, 28, 28) → (N, 784)
train_images = train_images.reshape(len(train_images), 784)#flattening of image 
test_images  = test_images.reshape(len(test_images), 784)

#print(train_images.shape, train_labels.shape)
#print(test_images.shape, test_labels.shape)'''
np.save('train_images.npy',train_images)
np.save('train_labels.npy',train_labels)
np.save('test_images.npy',test_images)
np.save('test_labels.npy',test_labels)