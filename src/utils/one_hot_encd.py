"""
Digit Recognition using MLP (from scratch)
Author: Atharv
Created: 2025
# Copyright (c) 2025 Atharv
# Licensed under GAPL 3.0
This implementation is written fully in NumPy for learning purposes.
"""
#run when brought fresh data or if you want to one hot encode something
#you need to load those labels first the encode and then save
import numpy as np
train_labels=np.load('train_labels.npy')
test_labels=np.load('test_labels.npy')
print(train_labels)
def one_hot(labels, num_classes=10):
    out = np.zeros((labels.size, num_classes), dtype=np.float32)
    labels=labels.astype(int)
    out[np.arange(labels.size), labels] = 1.0
    return out

train_labels=one_hot(train_labels)
test_labels=one_hot(test_labels)
np.save('train_labels.npy',train_labels)
np.save('test_labels.npy',test_labels)
