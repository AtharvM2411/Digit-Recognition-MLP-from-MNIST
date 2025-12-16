"""
Digit Recognition using MLP (from scratch)
Author: AtharvM2411
Created: 2025
# Copyright (c) 2025 AtharvM2411
# Licensed under GAPL 3.0
This implementation is written fully in NumPy for learning purposes.
"""
#ignore temp file made for debug purposes.
import numpy as np
from utils.fwd_pass import fwd_pass 
train_images=np.load('train_images.npy')
W3=np.load('W3.npy')
b3=np.load('b3.npy')
activation1, activation2, activation3_prob, z1, z2 = fwd_pass(train_images[:32])

print("Activation2 min/max:", activation2.min(), activation2.max())
print("Logits unique:", np.unique(activation2 @ W3 + b3))
print("Softmax output (first sample):", activation3_prob[0])
