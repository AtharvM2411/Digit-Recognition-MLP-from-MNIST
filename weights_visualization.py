import numpy as np
import matplotlib.pyplot as plt

W1 = np.load("W1.npy")
W2 = np.load("W2.npy")
W3 = np.load("W3.npy")
print(np.shape(W1),np.shape(W2),np.shape(W3))
fig = plt.figure(figsize=(15, 5))

# Create grids
x1 = np.arange(W1.shape[0])
y1 = np.arange(W1.shape[1])
X1, Y1 = np.meshgrid(x1, y1, indexing="ij")

x2 = np.arange(W2.shape[0])
y2 = np.arange(W2.shape[1])
X2, Y2 = np.meshgrid(x2, y2, indexing="ij")

x3 = np.arange(W3.shape[0])
y3 = np.arange(W3.shape[1])
X3, Y3 = np.meshgrid(x3, y3, indexing="ij")

# Subplot 1
ax1 = fig.add_subplot(131, projection="3d")
ax1.plot_surface(X1, Y1, W1)
ax1.set_title("W1")

# Subplot 2
ax2 = fig.add_subplot(132, projection="3d")
ax2.plot_surface(X2, Y2, W2)
ax2.set_title("W2")

# Subplot 3
ax3 = fig.add_subplot(133, projection="3d")
ax3.plot_surface(X3, Y3, W3)
ax3.set_title("W3")

plt.tight_layout()
plt.show()
