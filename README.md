# Balinese Script Image-to-Text Transliteration

This repository contains the code and experiments for a deep learning-based approach to transliterating Balinese palm-leaf manuscript images into Latin script. The project explores various image encoders and sequence decoders, incorporating both image-based and script-level data augmentation techniques.

## Features

- Word-level image-to-text transliteration
- Support for multiple encoder architectures (e.g., ResNet18, ViT, Swin Transformer)
- LSTM and GRU-based sequence decoders with attention
- Integration of image-level and script-level data augmentation
- Evaluation based on Character Error Rate (CER)
- Visualization tools for augmentations and results

## Dataset

This project uses the Balinese manuscript dataset provided by the ICFHR 2018 competition. The data includes word-level images and corresponding Latin transliterations. 

## How to Use

1. Clone the repository.
2. Set up the environment with the required dependencies.
3. Place the dataset in the correct folder.
4. Run the training script or use the provided notebooks to experiment.
