#!py -3.13
"""
Digit Recognition using MLP (from scratch)
Author: Atharv
Created: 2025
# Copyright (c) 2025 Atharv
# Licensed under GAPL 3.0
This implementation is written fully in NumPy for learning purposes.
"""
import numpy as np
import cv2
W1=np.load('W1.npy')
W2=np.load('W2.npy')
W3=np.load('W3.npy')
b1=np.load('b1.npy')
b2=np.load('b2.npy')
b3=np.load('b3.npy')
def fwd_pass(train_batched_random_images):
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
    return activation3_prob
# ===================== DRAWING PAD =====================

drawing = False       # True when mouse is pressed
last_x, last_y = None, None

canvas = np.zeros((300, 300), dtype=np.uint8)  # black canvas

def draw(event, x, y, flags, param):
    global drawing, last_x, last_y, canvas

    # Start drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_x, last_y = x, y

    # Stop drawing
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        last_x, last_y = None, None

    # Mouse move while drawing
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(canvas, (last_x, last_y), (x, y), 255, 20)  # thick white stroke
        last_x, last_y = x, y


cv2.namedWindow("Draw Digit (press 'c' to clear)")
cv2.setMouseCallback("Draw Digit (press 'c' to clear)", draw)

while True:
    # Show canvas
    show_canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    # ================= PREPROCESS FOR MNIST FORMAT =================
    kernel=(3,3)
    size_no_padding = (20,20)
    size = (28,28)
    #if to crop only if array is not empty
    if(drawing==True and np.any(canvas)):
        #find inked rows and cols
        inked_rows = np.any(canvas,axis=1)
        inked_cols = np.any(canvas,axis=0)
        #find indices of 1st and last inked cols and rows
        x_1st,x_last = np.where(inked_cols)[0][[0,-1]]
        y_1st,y_last = np.where(inked_rows)[0][[0,-1]]
        #slice the og image array
        digit=canvas[y_1st:y_last+1,x_1st:x_last+1]
    else:
        digit=canvas
    digit=cv2.resize(digit,size_no_padding,interpolation=cv2.INTER_AREA)#resize to no padding size with interpolation INTER_AREA
    digit = np.pad(digit,pad_width=(size[0]-size_no_padding[0])//2, mode='constant', constant_values=0)
    digit = cv2.GaussianBlur(digit,kernel,0.2)
    digit = digit / 255.0                      # normalize to 0–1
    digit = digit.reshape(1, size[0]*size[1])              # flatten

    # ================= PREDICT =================

    probs = fwd_pass(digit)
    if(np.max(probs)>0.4):
        pred = int(np.argmax(probs))
    else:
        pred="\0"

    # Draw prediction on the canvas window
    cv2.putText(show_canvas, f"Pred: {pred}", (10, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Draw Digit (press 'c' to clear)", show_canvas)

    key = cv2.waitKey(1)

    if key == ord('c'):          # clear canvas
        canvas[:] = 0
    elif key == ord('q'):        # quit
        break

cv2.destroyAllWindows()
