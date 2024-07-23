# MNIST Generative Adversarial Network (GAN) Project

This project is focused on developing a Generative Adversarial Network (GAN) trained on the MNIST dataset. The MNIST dataset consists of 70,000 images of handwritten digits, which our GAN will learn to generate new, synthetic images of handwritten digits that resemble the ones in the dataset.

## Project Overview

The goal of this project is to explore and understand the capabilities of GANs in generating new images that mimic the distribution of the MNIST dataset. This involves training a generator model to produce images that are indistinguishable from real MNIST images to the discriminator model, which tries to distinguish between real and generated images.

## Features

- **Data Preprocessing**: Scripts to preprocess the MNIST dataset for optimal training performance.
- **Model Architecture**: Detailed architecture for both the generator and discriminator models, designed specifically for the MNIST dataset.
- **Training Pipeline**: A comprehensive training pipeline that includes model training, validation, and logging of performance metrics.
- **Result Visualization**: Tools for visualizing the generated images and comparing them with real images from the MNIST dataset.
- **Model Evaluation**: Evaluation metrics and methods to assess the performance of the trained GAN.

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository:
`git clone https://github.com/neilSchroeder/MNIST-GAN.git`

2. Install the required dependencies:
```
conda create env -f environment.yml
conda activate mnist-gan
```
3. Run the training script:
`python train.py`

4. Explore the generated images in the `output` directory.

## Dependencies

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.