"""
Digit Recognition using MLP (from scratch)
Author: Atharv
Created: 2025
# Copyright (c) 2025 Atharv
# Licensed under GAPL 3.0
This implementation is written fully in NumPy for learning purposes.
"""
import numpy as np
activation3_prob=np.load('activation3_prob.npy')
activation2=np.load('activation2.npy')
batch_size=32
activation1=np.load('activation1.npy')
train_batched_random_labels=np.load('train_batched_random_labels.npy')
X=1#batch index
W2=np.load('W2.npy')
W3=np.load('W3.npy')
z1=np.load('z1.npy')
z2=np.load('z2.npy')
dL_dz3 = (activation3_prob - train_batched_random_labels) / batch_size  # shape (batch, 10)#loss function taken here ((activation3_prob - train_batched_random_labels) / batch_size) is explained in loss.py please do refer.

# a2 = activation from previous layer (Hidden2), shape (batch, H2)
dW3 = activation2.T @ dL_dz3           # shape (H2, 10)
db3 = np.sum(dL_dz3, axis=0, keepdims=True)  # shape (1, 10)

# dz2 = dL/da2 * da2/dz2
# If activation is ReLU: da2/dz2 = (z2 > 0).astype(float)
dz2 = (dL_dz3 @ W3.T) * (z2 > 0)

# a1 = activation from Hidden1, shape (batch, H1)
dW2 = activation1.T @ dz2       # shape (H1, H2)
db2 = np.sum(dz2, axis=0, keepdims=True)  # shape (1, H2)

# dz1 = dL/da1 * da1/dz1
dz1 = (dz2 @ W2.T) * (z1 > 0)  # ReLU derivative

# X = input batch, shape (batch, 784)
dW1 = X.T @ dz1         # shape (784, H1)
db1 = np.sum(dz1, axis=0, keepdims=True)  # shape (1, H1)