"""
Digit Recognition using MLP (from scratch)
Author: AtharvM2411
Created: 2025
# Copyright (c) 2025 AtharvM2411
# Licensed under GAPL 3.0
This implementation is written fully in NumPy for learning purposes.
"""
#for debug and understanding only
import numpy as np
activation3_prob=np.load('activation3_prob.npy')#p
train_batched_random_labels=np.load('train_batched_random_labels.npy')#y
batch_size=np.shape(activation3_prob)[0]
def loss_function_fwd(activation3_prob,train_batched_random_labels):
    activation3_prob=np.clip(activation3_prob,1e-12,1.0)
    loss_fwd=-np.log (np.sum(activation3_prob*train_batched_random_labels,axis=1))
    # p*y keeps only the predicted probability of the correct class because y is one-hot.
    # sum(...) extracts that probability (shape becomes (batch_size,)).
    # -log(prob_correct) is the cross-entropy loss for each sample.

    mean_loss_fwd=np.mean(loss_fwd)
    return mean_loss_fwd


def cross_entropy_backward(activation3_prob, train_batched_random_labels):
    activation3_prob = np.clip(activation3_prob, 1e-12, 1.0)
    batch_size = activation3_prob.shape[0]
    return - (train_batched_random_labels / activation3_prob) / batch_size#y/p is differentiation of fwd cross entropy loss

# dL/dz = gradient of loss w.r.t. logits (input to softmax)
# For softmax + cross-entropy, this simplifies to (p - y) / batch_size
# Explanation:
# 1. dL/dp = - y / p          (gradient of CE w.r.t softmax probabilities)
# 2. dp/dz = softmax Jacobian  (derivative of softmax)
# 3. Using chain rule: dL/dz = dL/dp * dp/dz
# 4. The math simplifies neatly to: dL/dz = p - y
# 5. Divide by batch_size because we take mean loss across the batch

# Softmax Jacobian J[i,j] = ∂p_i / ∂z_j for a single sample
# Diagonal entries (i == j):   J[i,i] = p_i * (1 - p_i)
# Off-diagonal entries (i != j): J[i,j] = -p_i * p_j
# This matrix shows how each output probability changes with respect to each input logit.
# In practice, we rarely compute this full Jacobian because
# softmax + cross-entropy simplifies the backward gradient to (p - y).
dL_dz3 = (activation3_prob - train_batched_random_labels) / batch_size