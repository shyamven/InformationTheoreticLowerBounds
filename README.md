# Information-Theoretic Lower Bounds

## Overview
This repository provides a Python implementation of an autoencoder designed to benchmark the relationship between empirical mean squared error (MSE) and an information-theoretic lower bound on generalization error. The code enables experiments for detecting overfitting and for streamlining neural architecture search by identifying architectures that cannot achieve a given performance threshold.

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
