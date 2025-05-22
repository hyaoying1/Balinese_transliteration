# Balinese Handwritten Word Transliteration

This repository provides the trained model and essential code for transliterating **Balinese handwritten word images** into **Latin characters**, using an encoder-decoder architecture with attention.

##  Model Architecture

The model follows a **segmentation-free, two-stage design**:

1. **Encoder**: A vision-based encoder (e.g., ResNet18, ViT, or hybrid CNN-ViT) extracts visual features from word images.
2. **Decoder**: A sequence-to-sequence model (LSTM or GRU) with attention decodes visual features into Latin character sequences.

## 📁 Files Included

- `your_model_name_best.pth`: Combined checkpoint containing:
  - Encoder and decoder weights
  - Optimizer states
  - Last epoch
  - Best validation loss

- `decoder.py`, `encoder.py`: Model definition files.
- `train.py`: Training pipeline and early stopping logic.
- `inference.py`: Inference and evaluation scripts (optional).
- `README.md`: You are here.

## How to Load the Model

```python
import torch
from encoder import YourEncoderClass
from decoder import YourDecoderClass

# Instantiate models
encoder = YourEncoderClass(...)
decoder = YourDecoderClass(...)

# Load checkpoint
checkpoint = torch.load('your_model_name_best.pth', map_location='cuda' or 'cpu')
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

# Optionally resume training
# optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
# epoch = checkpoint['epoch'] + 1
