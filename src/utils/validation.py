"""
Digit Recognition using MLP (from scratch)
Author: AtharvM2411
Created: 2025
# Copyright (c) 2025 AtharvM2411
# Licensed under GAPL 3.0
This implementation is written fully in NumPy for learning purposes.
"""
import numpy as np
from fwd_pass import fwd_pass
def compute_accuracy(activation3_prob, test_labels):
    """
    pred_probs : np.array of shape (batch_size, num_classes) → softmax output
    true_labels: np.array of shape (batch_size, num_classes) → one-hot
    """
    pred_classes = np.argmax(activation3_prob, axis=1)       # predicted class indices
    true_classes = np.argmax(test_labels, axis=1)      # true class indices
    accuracy = np.mean(pred_classes == true_classes)   # fraction correct
    return accuracy
# Load test data
test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')

# Forward pass on entire test set
activation1, activation2, activation3_prob, z1, z2 = fwd_pass(test_images)

# Compute accuracy
test_acc = compute_accuracy(activation3_prob, test_labels)
print(f"Test Accuracy: {test_acc*100:.2f}%")