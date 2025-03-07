# Information-Theoretic Lower Bounds

## Overview
This repository contains a Python implementation of an autoencoder for experimental analysis in the field of information theory. The autoencoder is designed to compare the mean squared error (MSE) with an information theoretic lower bound on generalization MSE across various datasets.

## Installation

To run this project, first clone the repository to your local machine: \
git clone https://github.com/your-username/your-repo-name.git \
cd your-repo-name

### Requirements

Ensure you have the following installed:
- Python 3
- NumPy
- PyTorch

You can install the necessary libraries using: \
pip install numpy torch

## Usage

The `main.py` script is used to run the simulations. You can specify various parameters as command-line arguments. \
python3 main.py --Dataset "DatasetName" [other options]

### Command-Line Arguments

- `--Dataset`: (Required) Name of the dataset to upload.
- `--TrainSize`: (Optional) Training dataset size. Default is 10000.
- `--BatchSize`: (Optional) Size of each training batch. Default is 100.
- `--Epochs`: (Optional) Number of training epochs. Default is 500.
- `--Iterations`: (Optional) Number of iterations to repeat the task. Default is 10.
- `--InputDim`: (Optional) Dimensionality of the input space. Default is 100.
- `--LatentDim`: (Optional) Dimensionality of the latent space. Default is 10.
- `--NumHidden`: (Optional) Number of hidden layers in the encoder/decoder. Default is 4.

## Results

The results of the simulations are saved in the `Results` directory as CSV files.
