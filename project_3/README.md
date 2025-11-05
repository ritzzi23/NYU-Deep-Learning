# Project 3: Vision Transformer (ViT)

This project implements a Vision Transformer from scratch, providing hands-on experience with transformer architectures applied to computer vision.

## Overview

The Vision Transformer (ViT) is a transformer-based architecture that treats images as sequences of patches, applying self-attention mechanisms to learn visual representations. This implementation builds ViT components from scratch without relying on pre-built transformer libraries.

## Architecture Components

### 1. PatchEmbedding

**Purpose**: Divides input images into non-overlapping patches and projects them into embedding space.

**Implementation**:
- Uses 2D convolution with kernel size and stride equal to patch size
- Converts image patches into linear embeddings
- Handles image preprocessing and patch extraction

**Key Features**:
- Configurable patch size
- Support for multi-channel images
- Efficient patch extraction using convolution

### 2. MultiHeadSelfAttention

**Purpose**: Implements the core attention mechanism with multiple attention heads.

**Implementation**:
- Query, Key, Value projections
- Scaled dot-product attention
- Multi-head attention with concatenation
- Dropout for regularization

**Key Features**:
- Configurable number of attention heads
- Scaled attention scores (dividing by sqrt(head_dim))
- Proper tensor reshaping for multi-head computation
- Output projection layer

### 3. Vision Transformer (ViT)

**Complete Architecture**:
- Patch embedding layer
- Class token for classification
- Positional embeddings
- Multiple transformer encoder blocks (MultiHeadSelfAttention + MLP)
- Layer normalization
- Classification head

**Training**:
- Trained on image classification task
- Uses standard cross-entropy loss
- Supports various image datasets

## Files

- `impl.ipynb`: Complete ViT implementation notebook
- `best_model.pth`: Trained model weights

## Implementation Details

### Key Design Choices

1. **Patch Extraction**: Uses convolution for efficient patch embedding
2. **Positional Encoding**: Adds learnable positional embeddings to patch tokens
3. **Class Token**: Prepends a learnable class token for classification
4. **Layer Normalization**: Applied before attention and MLP blocks
5. **Residual Connections**: Includes skip connections throughout

### Training Configuration

- Image size: Configurable (typically 224x224)
- Patch size: Configurable (typically 16x16)
- Embedding dimension: Configurable
- Number of layers: Configurable
- Number of attention heads: Configurable

## Usage

```python
# Example usage
from impl import VisionTransformer

model = VisionTransformer(
    image_size=224,
    patch_size=16,
    in_channels=3,
    embed_dim=768,
    num_layers=12,
    num_heads=12,
    num_classes=1000
)

# Load trained weights
model.load_state_dict(torch.load('best_model.pth'))
```

## Learning Outcomes

- Deep understanding of transformer architecture
- Self-attention mechanism implementation
- Vision transformer design principles
- Image classification with transformers
- Building complex architectures from scratch

## Key Concepts Learned

1. **Self-Attention**: How attention mechanisms work for visual data
2. **Patch-based Processing**: Treating images as sequences
3. **Positional Encoding**: Adding spatial information to transformers
4. **Multi-Head Attention**: Capturing different types of relationships
5. **Transfer Learning**: Adapting transformers for vision tasks

## Dependencies

- PyTorch
- torchvision
- NumPy
- Matplotlib
- PIL/Pillow

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- PyTorch Vision Transformer implementation
- Hugging Face timm library (for reference)

