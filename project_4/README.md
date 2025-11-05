# Project 4: Structured Prediction with CTC

This project implements structured prediction models for transcribing words from images, addressing the challenge of variable-length sequence prediction using Connectionist Temporal Classification (CTC).

## Overview

Structured prediction involves predicting structured outputs (like sequences of variable length) rather than fixed-size outputs. This project focuses on word transcription from images, where words have different lengths, making it a challenging sequence-to-sequence problem.

## Problem Statement

**Task**: Transcribe a word from an image containing text

**Challenges**:
- Variable-length outputs (different words have different lengths)
- Alignment between input and output sequences
- Need for special handling of blank characters and repetitions

## Architecture

### CTC (Connectionist Temporal Classification)

CTC is a loss function designed for sequence-to-sequence tasks where the input and output sequences have different lengths and alignment is unknown.

**Key Components**:

1. **Encoder**: Processes the input image
   - CNN backbone for feature extraction
   - Optional RNN/LSTM layers for temporal modeling

2. **Decoder/CTC Head**: Maps features to character predictions
   - Linear layer for character classification
   - Outputs probability distribution over alphabet + blank token

3. **Loss Function**: CTC loss
   - Handles alignment automatically
   - Allows multiple paths through the prediction space

### Alternative: GTN Framework

The project also compares models trained with CTC to models trained using the GTN (Graph Transformer Network) framework, which provides:
- Viterbi path finding
- Efficient training with weighted automata
- Alternative approach to sequence alignment

## Dataset

**Synthetic Word Dataset**:
- Generates random words from lowercase English alphabet
- Uses custom font rendering (Anonymous font)
- Supports data augmentation:
  - **Jitter**: Horizontal character shifting
  - **Noise**: Random noise injection

**Dataset Features**:
- Variable-length words
- Black and white images
- Configurable maximum word length

## Files

- `impl.ipynb`: Complete implementation notebook with CTC and GTN comparisons

## Implementation Details

### CTC Loss

The CTC loss function:
- Handles variable-length sequences
- Manages blank tokens and repetitions
- Uses dynamic programming for efficient computation
- Allows multiple valid alignments

### Model Architecture

```
Input Image (H x W)
    ↓
CNN Feature Extractor
    ↓
RNN/LSTM (optional, for temporal modeling)
    ↓
Linear Layer (alphabet_size + 1)  # +1 for blank token
    ↓
CTC Loss / GTN Loss
```

### Training Process

1. **Forward Pass**: Extract features and predict character probabilities
2. **Loss Computation**: Calculate CTC loss between predictions and ground truth
3. **Backward Pass**: Update model parameters
4. **Decoding**: Use CTC decoding (greedy or beam search) to get final transcription

## Usage

```python
# Example: Training a CTC model
from impl import CTCModel, CTCLoss

model = CTCModel(
    input_size=(32, 128),  # Image dimensions
    alphabet_size=27,      # 26 letters + blank
    hidden_dim=256
)

criterion = CTCLoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for image, target_text in dataloader:
    predictions = model(image)
    loss = criterion(predictions, target_text)
    loss.backward()
    optimizer.step()
```

## Key Concepts

### CTC Alignment

CTC solves the alignment problem by:
- Allowing blank tokens
- Collapsing repeated characters
- Finding the most likely alignment automatically

### Decoding Strategies

1. **Greedy Decoding**: Choose most likely character at each timestep
2. **Beam Search**: Maintain multiple hypotheses during decoding

### GTN Framework

GTN provides:
- Weighted finite automata for sequence modeling
- Efficient Viterbi path computation
- Alternative loss functions for sequence prediction

## Learning Outcomes

- Understanding structured prediction problems
- Implementation of CTC loss and decoding
- Handling variable-length sequences
- Sequence-to-sequence modeling
- Comparison of different frameworks (CTC vs GTN)
- Text recognition from images

## Dependencies

- PyTorch
- GTN (Graph Transformer Network) framework
- PIL/Pillow
- NumPy
- Matplotlib

## References

- [CTC: Connectionist Temporal Classification](https://distill.pub/2017/ctc/)
- [Weighted Automata in ML](https://awnihannun.com/writing/automata_ml/automata_in_machine_learning.pdf)
- [GTN Documentation](https://gtn.readthedocs.io/en/latest/)

