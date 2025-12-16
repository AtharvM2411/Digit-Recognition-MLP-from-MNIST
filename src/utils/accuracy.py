"""
Digit Recognition using MLP (from scratch)
Author: AtharvM2411
Created: 2025
# Copyright (c) 2025 AtharvM2411
# Licensed under GAPL 3.0
This implementation is written fully in NumPy for learning purposes.
"""
import numpy as np
def compute_accuracy(pred_probs, true_labels):
    """
    pred_probs : np.array of shape (batch_size, num_classes) → softmax output
    true_labels: np.array of shape (batch_size, num_classes) → one-hot
    """
    pred_classes = np.argmax(pred_probs, axis=1)       # predicted class indices
    true_classes = np.argmax(true_labels, axis=1)      # true class indices
    accuracy = np.mean(pred_classes == true_classes)   # fraction correct
    return accuracy
