import numpy as np
import matplotlib.pyplot as plt

"""
Digit Recognition using MLP (from scratch)
Author: Atharv
Created: 2025
# Copyright (c) 2025 Atharv
# Licensed under GAPL 3.0
This implementation is written fully in NumPy for learning purposes.
"""

#_______________________________________________
#Loading data
#_______________________________________________
W1=np.load('w1.npy')
W2=np.load('w2.npy')
W3=np.load('w3.npy')
b1=np.load('b1.npy')
b2=np.load('b2.npy')
b3=np.load('b3.npy')
#________________________________________________
#Batching
#________________________________________________
train_images=np.load('train_images.npy')#load data
train_labels=np.load('train_labels.npy')#load labels
#--------------------------------------------------------------------------------------------------------------------
batch_size = 32#set as per convenience, try to keep this as a factor of total number of data
#---------------------------------------------------------------------------------------------------------------------
def batching(train_images,train_labels,batch_size):#defining function of batching
    indices=np.arange(np.shape(train_images)[0])#making array of indices for each sample in dataset
    np.random.shuffle(indices)#shuffling the prev made array
    
    for i in range(0,len(indices),batch_size):#looping over indices with step=batch_size
        t=indices[i:i+batch_size]#slicing indices array 
        #taking indices[i : i + batch_size] gives you the next batch_size random samples.
        # Example: if indices = [5, 12, 3, 9, 0, 20, ...] and batch_size = 2,
        # then indices[0:2] → [5, 12], indices[2:4] → [3, 9], etc.

        train_batch_of_random_images=train_images[t]#1 batch images 
        train_batch_of_random_labels=train_labels[t]#1 batch labels

        yield train_batch_of_random_images,train_batch_of_random_labels

batches = list(batching(train_images, train_labels, batch_size))

    # extract only images into numpy array
    # 'batch[0]' is the image batch inside the tuple (images, labels).
    # We collect them into a list, and then convert that list into a NumPy array.
    # Final shape will be: (num_batches, batch_size, 784)
train_batched_random_images = np.array([batch[0] for batch in batches])

    # extract only labels into numpy array
    # 'batch[1]' is the label batch inside the tuple.
    # Shape will become: (num_batches, batch_size, 10) for one-hot labels,
    # or (num_batches, batch_size) for integer labels.
train_batched_random_labels = np.array([batch[1] for batch in batches])

np.save('train_batched_random_images.npy',train_batched_random_images)
np.save('train_batched_random_labels.npy',train_batched_random_labels)


#________________________________________________
#forward pass
#________________________________________________
def fwd_pass(train_batched_random_images):
    #train_batched_random_images=np.load('train_batched_random_images.npy')
    #train_batched_random_labels=np.load('train_batched_random_labels.npy')
    def softmax(x):
        # subtract max for numerical stability, per row
        x = x - np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x)
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    z1=train_batched_random_images @ W1 + b1
    activation1=np.maximum(0,z1)# 1st layer Computation with ReLU
    activation1 = np.clip(activation1, 0, 100)#clip to avoide out of bound
    z2=activation1 @ W2 + b2
    activation2=np.maximum(0,z2)# 2nd layer Computation with ReLU
    activation2 = np.clip(activation2, 0, 100)
    z3=activation2 @ W3 + b3
    z3 = np.clip(z3, -500, 500)
    activation3_prob=softmax(z3)# 3rd layer Computation with Softmax, gives out probabilities

    #np.save('activation1.npy',activation1)
    #np.save('activation2.npy',activation2)
    #np.save('activation3_prob.npy',activation3_prob)
    return activation1,activation2,activation3_prob,z1,z2
    

#__________________________________________________________
#backward pass
#__________________________________________________________
def back_prop(activation1,activation2,activation3_prob,W1,W2,W3,b1,b2,b3,train_batched_random_labels,X,z1,z2):
    
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
    return dW1,db1,dW2,db2,dW3,db3
#___________________________________________
#Update
#___________________________________________
def update(W1, b1, W2, b2, W3,b3,dW1, db1, dW2, db2,dW3,db3,lr):
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    W3 -= lr * dW3
    b3 -= lr * db3
    np.save('W1.npy',W1)
    np.save('W2.npy',W2)
    np.save('W3.npy',W3)
    np.save('b1.npy',b1)
    np.save('b2.npy',b2)
    np.save('b3.npy',b3)
    return W1,b1,W2,b2,W3,b3

#__________________________________________________________________
# TRAINING LOOP
#__________________________________________________________________

epochs = 100
lr = 0.001
losses = []
accuracy=[]
for epoch in range(epochs):
    print(f"\n===== EPOCH {epoch+1}/{epochs} =====")

    # regenerate new random batches each epoch
    batches = list(batching(train_images, train_labels, batch_size))

    total_loss = 0

    for batch_idx, (X, y) in enumerate(batches):#enum gives paired iterable (0,seq[0]),(1,seq[1]),...

        # ---------- FORWARD PASS ----------
        activation1, activation2, activation3_prob, z1, z2 = fwd_pass(X)

        # compute batch loss
        # loss = -log(correct_prob)
        correct_probs = np.sum(activation3_prob * y, axis=1)
        loss = -np.log(correct_probs + 1e-12)
        batch_loss = np.mean(loss)
        losses.append(batch_loss)#appends batch loss in loss list to plot later 
        total_loss += batch_loss
        

        # ---------- BACKWARD PASS ----------
        dW1, db1, dW2, db2, dW3, db3 = back_prop(
            activation1, activation2, activation3_prob,
            W1, W2, W3, b1, b2, b3,
            y, X, z1, z2
        )

        # ---------- PARAMETER UPDATE ----------
        W1, b1, W2, b2, W3, b3 = update(
            W1, b1, W2, b2, W3, b3,
            dW1, db1, dW2, db2, dW3, db3,
            lr
        )
        pred_class = np.argmax(activation3_prob, axis=1)   # predicted class index
        true_class = np.argmax(y, axis=1)                  # ground truth class index
        for i in range(len(y)):
            if(pred_class[i]==true_class[i]):
                accuracy.append(1)
            else:
                accuracy.append(0)
        # print progress occasionally
        if batch_idx % 100 == 0:
            #accuracy-------------------------------------------------------------- accuracy
            pred_classes = np.argmax(activation3_prob, axis=1)   # predicted class index
            true_classes = np.argmax(y, axis=1)                  # ground truth class index

            batch_accuracy = np.mean(pred_classes == true_classes)
            print(f"Batch {batch_idx}/{len(batches)} | Loss = {batch_loss:.4f} | Accuracy:{batch_accuracy*100}")

    print(f"Epoch {epoch+1} completed. Mean Loss = {total_loss/len(batches):.4f}  Accuracy={np.mean(accuracy)*100}")
plt.plot(losses[:len(losses):100])
plt.xlabel('Iteration / Batch')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training Loss over Time')
plt.show()

