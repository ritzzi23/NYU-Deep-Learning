# Project 2: Deep Learning Architectures

This project explores various deep learning architectures through practical implementations and experiments.

## Overview

Project 2 consists of four main notebooks covering different neural network architectures and applications:
1. **CNN**: Convolutional Neural Networks for image classification
2. **RNN**: Recurrent Neural Networks for sequence modeling
3. **MoE**: Mixture of Experts model
4. **Sequence Classification**: Sequence classification tasks

## Components

### 1. CNN (`cnn.ipynb`)

**Task**: Cats vs Dogs Image Classification

**Description**: 
- Implements and trains convolutional neural networks to classify images of cats and dogs
- Uses the Cats and Dogs filtered dataset from TensorFlow
- Explores CNN architectures, data augmentation, and visualization techniques

**Key Features:**
- Data loading and preprocessing
- CNN architecture design
- Training loop implementation
- Model evaluation and visualization
- Intermediate layer activation visualization

**Dataset**: 
- Training: 1000 cats + 1000 dogs
- Validation: 500 cats + 500 dogs

### 2. RNN (`rnn.ipynb`)

**Task**: Character-level Sequence Modeling

**Description**:
- Implements RNN models for sequence prediction
- Works with sequences of integers (0-26, representing alphabet letters)
- Similar to echo data tasks but extended to multi-class character prediction

**Key Features:**
- RNN/LSTM/GRU implementations
- Sequence-to-sequence modeling
- Character-level prediction
- Training on sequential data

**Learning Objectives:**
- Understanding recurrent architectures
- Handling variable-length sequences
- Memory mechanisms in neural networks

### 3. Mixture of Experts (`moe.ipynb`)

**Task**: Implementing a Mixture of Experts Model

**Description**:
- Implements an MoE architecture with expert networks and a gating network
- Trains on a synthetic dataset with decision boundaries
- Visualizes expert specialization and gating behavior

**Components Implemented:**
- **Expert Model**: Simple neural network with one linear layer
- **Gating Network**: Outputs probabilities for choosing each expert
- **MoE Model**: Combines experts using gating network predictions
- **Training Loop**: Includes learning rate scheduling

**Key Features:**
- Binary cross-entropy loss
- Adam optimizer with learning rate decay
- Decision boundary visualization
- Expert specialization analysis

**Bonus Task**: Extended MoE with additional experts and enhanced visualization

### 4. Sequence Classification (`seq_classification.ipynb`)

**Task**: Sequence Classification Tasks

**Description**:
- Implements models for sequence classification problems
- Explores different architectures for handling sequential data
- Applies to various sequence classification tasks

## Files

- `cnn.ipynb`: Convolutional Neural Networks notebook
- `rnn.ipynb`: Recurrent Neural Networks notebook
- `moe.ipynb`: Mixture of Experts notebook
- `seq_classification.ipynb`: Sequence Classification notebook

## Usage

Each notebook is self-contained and can be run independently. They are designed to work in Google Colab with GPU support.

To run:
1. Open the notebook in Google Colab
2. Enable GPU: Runtime -> Change Runtime Type -> GPU
3. Run all cells sequentially

## Learning Outcomes

- Understanding of CNN architectures and their applications
- Hands-on experience with RNNs and sequence modeling
- Implementation of ensemble methods (MoE)
- Practical experience with image classification
- Sequence classification techniques
- Model visualization and interpretation

## Dependencies

- PyTorch
- torchvision
- NumPy
- Matplotlib
- PIL/Pillow
- Google Colab (for GPU access)

