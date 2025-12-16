"""
Digit Recognition using MLP (from scratch)
Author: AtharvM2411
Created: 2025
# Copyright (c) 2025 AtharvM2411
# Licensed under GAPL 3.0
This implementation is written fully in NumPy for learning purposes.
"""
import numpy as np
def fwd_pass(train_batched_random_images):
    W1=np.load('W1.npy')
    W2=np.load('W2.npy')
    W3=np.load('W3.npy')
    b1=np.load('b1.npy')
    b2=np.load('b2.npy')
    b3=np.load('b3.npy')
    #train_batched_random_images=np.load('train_batched_random_images.npy')
    def softmax(x):
        # subtract max for numerical stability, per row
        x = x - np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x)
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    z1=train_batched_random_images @ W1 + b1# 1st layer Computation with ReLU
    activation1=np.maximum(0,z1)
    z2=activation1 @ W2 + b2# 2nd layer Computation with ReLU
    activation2=np.maximum(0,z2)
    activation3_prob=softmax(activation2 @ W3 + b3)# 3rd layer Computation with Softmax, gives out probabilities

    print(activation3_prob)
    #np.save('activation1.npy',activation1)
    #np.save('activation2.npy',activation2)
    #np.save('activation3_prob.npy',activation3_prob)
    return activation1, activation2, activation3_prob, z1, z2