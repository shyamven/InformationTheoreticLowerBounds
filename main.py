import argparse
import sys
import numpy as np
import torch
import os

sys.path.append('./InformationTheoreticLowerBounds')
from utils.upload_data import LoadDataset
from models.train_autoencoder import train_encoder_results
# to run file: python3 main.py --Dataset "DatasetName" (optional: --Shift "Shift", etc.)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run simulations.')
    parser.add_argument('--Dataset', type=str, help='Dataset to upload', required=True)
    parser.add_argument('--TrainSize', type=int, default=10000, help="Training dataset size", required=False)
    parser.add_argument('--BatchSize', type=int, default=100, help='Size of each training batch')
    parser.add_argument('--Epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--Iterations', type=int, default=10, help='Number of iterations to repeat task')
    parser.add_argument('--InputDim', type=int, default=100, help='Dimensionality of the input space')
    parser.add_argument('--LatentDim', type=int, default=10, help='Dimensionality of the latent space')
    parser.add_argument('--NumHidden', type=int, default=2, help='Number of hidden layers in the encoder/decoder')
    args = parser.parse_args()
    
    DatasetName = args.Dataset
    train_size = args.TrainSize
    batch_size = args.BatchSize
    epochs = args.Epochs
    iterations = args.Iterations
    input_dim = args.InputDim
    latent_dim = args.LatentDim
    num_hidden = args.NumHidden

    # This function calls the dataset we need for the experiment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, X_test = LoadDataset(DatasetName, train_size, input_dim)
    
    # Compare Autoencoder MSE With Lower Bound
    train_losses, test_losses, lower_bounds = train_encoder_results(DatasetName, X_train, X_test, iterations, epochs, batch_size, input_dim, latent_dim, num_hidden, device)
    
    # Path for the directory
    results_dir = 'Results'
    
    # Check if the directory exists, and if not, create it
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    # Path for the CSV files
    train_losses_path = f'Results/TRAIN_{DatasetName}_InputDim={input_dim}_LatentDim={latent_dim}_NumHidden={num_hidden}_TrainSize={train_size}_BatchSize={batch_size}_Epochs={epochs}_Iterations={iterations}.csv'
    test_losses_path = f'Results/TEST_{DatasetName}_InputDim={input_dim}_LatentDim={latent_dim}_NumHidden={num_hidden}_TrainSize={train_size}_BatchSize={batch_size}_Epochs={epochs}_Iterations={iterations}.csv'
    lower_bounds_path = f'Results/LB_{DatasetName}_InputDim={input_dim}_LatentDim={latent_dim}_NumHidden={num_hidden}_TrainSize={train_size}_BatchSize={batch_size}_Epochs={epochs}_Iterations={iterations}.csv'
    
    # Save matrices to CSV
    np.savetxt(train_losses_path, train_losses, delimiter=',')
    np.savetxt(test_losses_path, test_losses, delimiter=',')
    np.savetxt(lower_bounds_path, lower_bounds, delimiter=',')
