"""
Digit Recognition using MLP (from scratch)
Author: Atharv
Created: 2025
# Copyright (c) 2025 Atharv
# Licensed under GAPL 3.0
This implementation is written fully in NumPy for learning purposes.
"""
#run once only for training from start (randomizes the weights)
import numpy as np
H1=128#no of neurons in 1st layer
H2=64#no of neurons in 2st layer
otp=10#no of neurons in output layer
def wt_init_output(inc,otg):# xavier initialization
    a=np.sqrt(6/inc+otg)
    w=np.random.uniform(-a,a,(inc,otg))
    return w
def wt_init_(inc,otg):#he intialization for relu
    W = np.random.randn(inc, otg) * np.sqrt(2/inc)
    return W
W1=wt_init_(784,H1)
W2=wt_init_(H1,H2)
W3=wt_init_output(H2,otp)
np.save('W1.npy',W1)
np.save('W2.npy',W2)
np.save('W3.npy',W3)