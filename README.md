# Digit-Recognition-MLP-MNIST
This Project is an MLP trained to recognize digits from 0 to 9 using only NumPy in Python.

---

# Digit Recognition using MLP (From Scratch – NumPy)

This project implements a **Multi-Layer Perceptron (MLP)** for handwritten digit recognition using the **MNIST dataset**, written **Using NumPy only** (no PyTorch / TensorFlow).

The project is designed for **learning and experimentation**, with a strong focus on:

* Understanding forward propagation and backpropagation mathematically
* Manual batching, loss computation, and gradient updates
* Real-time inference using a drawing pad
* Try to preserve the naming convention of project if you dont want to change the script multiple times. 

> **Note**
> Support for **AMNIST (Augmented MNIST – 400k samples)** is **planned for future versions**.
> The current training pipeline is fully stable for **original MNIST**.
> Real time Inference using cv2 is available using weights and biases in main folder.
> Weight Visualization script is built in project folder.
---

## Project Structure

```
Digit_Recognition_MLP_MNIST
│
├── W1.npy, W2.npy, W3.npy          # Trained weights
├── b1.npy, b2.npy, b3.npy          # Trained biases
│
├── train_images.npy                # (N, 784)
├── train_labels.npy                # (N, 10) one-hot
├── test_images.npy
├── test_labels.npy
│
├── train_batched_random_images.npy
├── train_batched_random_labels.npy
│
├── real_time_testing.py             # Drawing pad inference
│
├── weights_visualization.py
│
├── model_init/
│   ├── weights_init.py
│   └── biases_init.py
│
├── data/
│   └── mnist/
│       ├── train-images.idx3-ubyte
│       ├── train-labels.idx1-ubyte
│       ├── t10k-images.idx3-ubyte
│       └── t10k-labels.idx1-ubyte
│
├── src/
│   ├── main_training_logic.py
│   ├── data_condition_main_og_mnist.py
│   ├── data_conditioning_main_aug_mnist.py   # AMNIST – future support
│   ├── debug.py
│   │
│   └── utils/
│       ├── batching.py
│       ├── fwd_pass.py
│       ├── back_prop.py
│       ├── loss.py
│       ├── accuracy.py
│       ├── testing.py
│       ├── one_hot_encd.py
│       └── backpropagation_explanation.txt
```

---

## Model Architecture
> This can bechanged through Updating Model_Init files.
> Number of Layers can be changed but it is integrated deeply in main_training_logic, backprop_logic and testing
> and will require changing all these Scripts
```
Input  → Hidden 1 → Hidden 2 → Output
784    →   128     →   64     →  10
```

| Component | Shape      |
| --------- | ---------- |
| Input     | (B, 784)   |
| W1        | (784, 128) |
| b1        | (1, 128)   |
| W2        | (128, 64)  |
| b2        | (1, 64)    |
| W3        | (64, 10)   |
| b3        | (1, 10)    |

Activation: ReLU
Output: Softmax
Loss: Categorical Cross-Entropy

---

## Forward Pass (with Shapes)

Let:

* `X` = input batch, shape `(B, 784)`
* `Y` = one-hot labels, shape `(B, 10)`

### Layer 1

```
Z1 = X · W1 + b1        → (B, 128)
A1 = ReLU(Z1)          → (B, 128)
```

### Layer 2

```
Z2 = A1 · W2 + b2      → (B, 64)
A2 = ReLU(Z2)          → (B, 64)
```

### Output Layer

```
Z3 = A2 · W3 + b3      → (B, 10)
ŷ  = Softmax(Z3)       → (B, 10)
```

---

## Loss Function

Categorical Cross-Entropy
Explanation Document is attached in folder.
For one sample:

```
L = − Σ yᵢ log(ŷᵢ)
```

For a batch:

```
L_batch = − (1/B) Σ Σ yᵢ log(ŷᵢ)
```

---

## Backpropagation (with Derivatives and Sizes)
Explanation Documentation is attached in folder.
### Output Layer (Softmax + Cross-Entropy)

Key simplification:

```
∂L / ∂Z3 = ŷ − Y
```

Shape:

```
dZ3 → (B, 10)
```

Gradients:

```
dW3 = (A2ᵀ · dZ3) / B   → (64, 10)
db3 = sum(dZ3) / B     → (1, 10)
```

---

### Hidden Layer 2

```
dA2 = dZ3 · W3ᵀ               → (B, 64)
dZ2 = dA2 ⊙ ReLU'(Z2)         → (B, 64)

dW2 = (A1ᵀ · dZ2) / B         → (128, 64)
db2 = sum(dZ2) / B           → (1, 64)
```

---

### Hidden Layer 1

```
dA1 = dZ2 · W2ᵀ               → (B, 128)
dZ1 = dA1 ⊙ ReLU'(Z1)         → (B, 128)

dW1 = (Xᵀ · dZ1) / B          → (784, 128)
db1 = sum(dZ1) / B           → (1, 128)
```

---

## Parameter Update (SGD)

```
W ← W − η · dW
b ← b − η · db
```

Gradients must be averaged **only once per batch**.
Double normalization will severely reduce learning.

---

## Training Pipeline (Correct Order)

1. Initialize parameters

   ```
   model_init/weights_init.py
   model_init/biases_init.py
   ```

2. Prepare dataset

   * Original MNIST (fully supported)

     ```
     src/data_condition_main_og_mnist.py
     ```
   * Augmented MNIST (AMNIST – future support)

     ```
     src/data_conditioning_main_aug_mnist.py
     ```

3. Train the model

   ```
   src/main_training_logic.py
   ```

4. Evaluate accuracy

   ```
   src/utils/testing.py
   ```

5. Real-time inference

   ```
   real_time_testing.py
   ```

6. Weights Visualization
   ```
   weights_visualization.py
   ```
---

## Real-Time Drawing Pad

* Uses OpenCV canvas
* Mouse-based digit drawing
* Input is:

  * Converted to grayscale
  * Centered and resized to 28×28
  * Normalized to [0, 1]

Real-time accuracy is typically **lower than MNIST test accuracy** due to:

* Stroke thickness variation
* Centering mismatch
* Contrast differences

---

## Common Issues and Debugging Insights

| Symptom                | Likely Cause                        |
| ---------------------- | ----------------------------------- |
| Loss stuck near 2.3    | Model guessing randomly             | [SOLVED]
| Softmax outputs ~0.1   | Dead gradients or normalization bug | [SOLVED]
| Same digit predicted   | Input preprocessing mismatch        |
| Low real-time accuracy | Drawing scale mismatch              |
| Training stalls        | Gradients averaged twice            | [SOLVED]

---

## AMNIST (Augmented MNIST) – Planned Support

* Dataset size: ~400k samples
* Same 28×28 grayscale format
* Increased data diversity

> AMNIST preprocessing scripts exist, but **full training and benchmarking support will be added in a future update** after stability validation.

---

## Possible Extensions

* Full AMNIST integration
* EMNIST (digits + letters)
* Deeper MLP or CNN
* Adam optimizer
* Online learning from drawing pad
* Gradient checking utilities

---

## Author Notes

This project is intentionally framework-free to build **deep intuition** about:

* Matrix calculus
* Backpropagation mechanics
* Training instability and debugging

By AtharvM2411