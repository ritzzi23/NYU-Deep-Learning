# NYU Deep Learning Course - Assignments Repository

This repository contains all assignments and projects from the NYU Deep Learning course (CSCI-GA 2572), covering fundamental concepts in deep learning from neural networks basics to advanced architectures.

## Repository Structure

```
NYU-Deep-Learning/
├── project_1/          # Neural Networks from Scratch
├── project_2/          # Deep Learning Architectures (CNN, RNN, MoE)
├── project_3/          # Vision Transformer (ViT)
├── project_4/          # Structured Prediction with CTC
└── Jepa-final_project/ # Final Project: JEPA World Model
```

## Projects Overview

### [Project 1: Neural Networks from Scratch](./project_1/)

**Topics**: MLP implementation, manual backpropagation, gradient descent visualization

- Implemented a Multi-Layer Perceptron (MLP) from scratch with manual forward and backward propagation
- Implemented MSE and BCE loss functions
- Created gradient descent visualization using VGG13 to synthesize images

**Key Learning**: Understanding the fundamentals of neural network mechanics without automatic differentiation

### [Project 2: Deep Learning Architectures](./project_2/)

**Topics**: CNNs, RNNs, Mixture of Experts, Sequence Classification

- **CNN**: Cats vs Dogs image classification using convolutional neural networks
- **RNN**: Character-level sequence modeling with recurrent architectures
- **MoE**: Mixture of Experts model with expert networks and gating mechanisms
- **Sequence Classification**: Various sequence classification tasks

**Key Learning**: Hands-on experience with major deep learning architectures

### [Project 3: Vision Transformer (ViT)](./project_3/)

**Topics**: Transformer architecture, Self-attention, Vision transformers

- Implemented Vision Transformer from scratch
- Built PatchEmbedding, MultiHeadSelfAttention, and complete ViT architecture
- Trained on image classification tasks

**Key Learning**: Understanding transformer architectures applied to computer vision

### [Project 4: Structured Prediction with CTC](./project_4/)

**Topics**: Structured prediction, CTC loss, Sequence-to-sequence models

- Implemented Connectionist Temporal Classification (CTC) for word transcription
- Compared CTC with GTN (Graph Transformer Network) framework
- Handled variable-length sequence prediction

**Key Learning**: Dealing with structured outputs and sequence alignment problems

### [Final Project: JEPA World Model](./Jepa-final_project/Deep-Learning-CSCI-GA-2572-Final-Project/)

**Topics**: Joint Embedding Predictive Architecture, Self-supervised learning, World models

- Implemented and trained a JEPA (Joint Embedding Predictive Architecture) world model
- Trained on 2.5M frames of agent trajectories in a two-room environment
- Evaluated learned representations through probing tasks
- Explored various regularization techniques to prevent representation collapse

**Key Learning**: Advanced self-supervised learning and world model architectures

## Getting Started

Each project folder contains:
- Implementation code (`.py` files or `.ipynb` notebooks)
- Project-specific README with detailed descriptions
- Requirements and dependencies

Refer to individual project READMEs for:
- Detailed descriptions
- Usage instructions
- Key concepts learned
- Dependencies

## Course Information

- **Course**: CSCI-GA 2572 - Deep Learning
- **Institution**: New York University (NYU)
- **Focus**: Practical implementation of deep learning concepts and architectures

## Technologies Used

- PyTorch
- NumPy
- Computer Vision (OpenCV, PIL)
- Various deep learning architectures (CNN, RNN, Transformer, JEPA)
- Specialized frameworks (GTN for structured prediction)

## Learning Journey

This repository demonstrates progression from:
1. **Fundamentals**: Manual implementation of basic neural networks
2. **Architectures**: Standard deep learning models (CNN, RNN)
3. **Advanced Architectures**: Transformers and attention mechanisms
4. **Specialized Applications**: Structured prediction, world models
5. **Research**: Final project exploring cutting-edge architectures

---

**Note**: This repository is for educational purposes, showcasing implementations from the NYU Deep Learning course.