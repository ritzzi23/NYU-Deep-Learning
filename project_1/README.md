# Project 1: Neural Networks from Scratch

This project implements fundamental neural network components from scratch, focusing on understanding the mechanics of forward and backward propagation.

## Overview

This project consists of two main components:
1. **MLP Implementation**: A Multi-Layer Perceptron with manual forward and backward propagation
2. **Gradient Descent Visualization**: Synthesizing images using gradient descent on a pre-trained VGG13 model

## Components

### 1. MLP (`mlp.py`)

A complete implementation of a 2-layer Multi-Layer Perceptron without using PyTorch's automatic differentiation.

**Features:**
- **Forward Pass**: Implements linear transformations with configurable activation functions (ReLU, Sigmoid, Identity)
- **Backward Pass**: Manual gradient computation using chain rule
- **Loss Functions**: 
  - Mean Squared Error (MSE) loss
  - Binary Cross-Entropy (BCE) loss

**Key Implementation Details:**
- Manual computation of gradients for weights and biases
- Proper handling of activation function derivatives
- Caching intermediate values for efficient backward pass

### 2. Gradient Descent (`gd.py`)

Demonstrates gradient-based image synthesis using a pre-trained VGG13-BN model.

**Features:**
- **Image Synthesis**: Generates images that maximize specific class predictions in VGG13
- **Data Augmentation**: Implements normalization and random jittering
- **Optimization Techniques**:
  - Gradient ascent with learning rate scheduling
  - Gaussian blur regularization to reduce high-frequency noise
  - Weight decay for regularization
  - Pixel value clamping to maintain valid image range

**Implementation Highlights:**
- Generates images for classes 0, 12, and 954 from ImageNet
- Uses gradient ascent (instead of descent) to maximize class predictions
- Applies Gaussian blur to gradients to improve visual quality
- Implements proper normalization using ImageNet statistics

## Files

- `mlp.py`: Complete MLP implementation with forward/backward pass
- `gd.py`: Gradient descent visualization using VGG13

## Usage

### MLP Training Example
```python
from mlp import MLP, mse_loss, bce_loss

# Initialize MLP
model = MLP(
    linear_1_in_features=784,
    linear_1_out_features=128,
    f_function='relu',
    linear_2_in_features=128,
    linear_2_out_features=10,
    g_function='identity'
)

# Forward pass
y_hat = model.forward(x)
loss, dJdy_hat = mse_loss(y, y_hat)

# Backward pass
model.backward(dJdy_hat)
```

### Gradient Descent Visualization
```python
python gd.py
```

This will generate synthesized images (`img_0.jpg`, `img_12.jpg`, `img_954.jpg`) that maximize predictions for the specified ImageNet classes.

## Learning Outcomes

- Understanding of forward and backward propagation mechanics
- Manual gradient computation using chain rule
- Implementation of common activation functions and their derivatives
- Gradient-based optimization techniques
- Image synthesis using pre-trained models

## Dependencies

- PyTorch
- NumPy
- OpenCV (cv2)
- torchvision
- tqdm

