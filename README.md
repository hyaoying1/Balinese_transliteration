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

## Structure

- `dataset/` – Data loading and preprocessing scripts  
- `models/` – Encoder and decoder model definitions  
- `augmentations/` – Image-based and script-level augmentation methods  
- `training/` – Training loops and evaluation functions  
- `notebooks/` – Jupyter notebooks for testing and visualization  
- `utils/` – Utility functions for logging, saving models, and metrics  

## How to Use

1. Clone the repository.
2. Set up the environment with the required dependencies.
3. Place the dataset in the correct folder.
4. Run the training script or use the provided notebooks to experiment.

## Citation

If you find this work useful, please cite the related ICFHR 2018 dataset and the script-level augmentation method from ICFHR 2022.

## License

This project is for research and educational purposes only.

---

Let me know if you'd like to include your name, a requirements.txt file, or links to specific papers.
