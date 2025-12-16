"""
Digit Recognition using MLP (from scratch)
Author: Atharv
Created: 2025
# Copyright (c) 2025 Atharv
# Licensed under GAPL 3.0
This implementation is written fully in NumPy for learning purposes.
"""
#run once only for training from start (resets the biases to zero)
import numpy as np
H1=128#no of neurons in 1st layer
H2=64#no of neurons in 2st layer
otp=10#no of neurons in output layer
b1 = np.zeros((1, H1))
b2 = np.zeros((1, H2))
b3 = np.zeros((1, otp))
np.save('b1.npy',b1)
np.save('b2.npy',b2)
np.save('b3.npy',b3)