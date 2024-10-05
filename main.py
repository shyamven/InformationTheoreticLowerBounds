import argparse
import sys
import numpy as np
import torch
import os

sys.path.append('./InformationTheoreticLowerBounds')
from utils.upload_data import LoadDataset, LoadDatasetReg
from models.train_autoencoder import train_encoder_results
from models.train_regression import train_regressor_results
# to run file: python3 main.py --Dataset "DatasetName" (optional: --Shift "Shift", etc.)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run simulations.')
    parser.add_argument('--Dataset', type=str, help='Dataset to upload', required=True)
    parser.add_argument('--Task', type=str, default='Autoencoder', help='Task (Regression or Autoencoder)')
    parser.add_argument('--LossFunction', type=str, help='Loss function (MSE, MSEL2)', required=True)
    parser.add_argument('--TrainSize', type=int, default=10000, help="Training dataset size", required=False)
    parser.add_argument('--BatchSize', type=int, default=100, help='Size of each training batch')
    parser.add_argument('--Epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--Iterations', type=int, default=10, help='Number of iterations to repeat task')
    parser.add_argument('--InputDim', type=int, default=100, help='Dimensionality of the input space')
    parser.add_argument('--LatentDim', type=int, default=10, help='Dimensionality of the latent space')
    parser.add_argument('--OutputDim', type=int, default=1, help='Dimensionality of the output space')
    parser.add_argument('--NumHidden', type=int, default=2, help='Number of hidden layers in the encoder/decoder')
    args = parser.parse_args()
    
    dataset_name = args.Dataset
    task = args.Task
    loss_function = args.LossFunction
    train_size = args.TrainSize
    batch_size = args.BatchSize
    epochs = args.Epochs
    iterations = args.Iterations
    input_dim = args.InputDim
    latent_dim = args.LatentDim
    output_dim = args.OutputDim
    num_hidden = args.NumHidden
    
    # Path for the directory
    results_dir = 'Results'
    
    # Check if the directory exists, and if not, create it
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # This function calls the dataset we need for the experiment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if task == 'Autoencoder':
        X_train, X_test, h_x = LoadDataset(dataset_name, train_size, input_dim)
        # Compare Autoencoder MSE With Lower Bound
        train_losses, test_losses, lower_bounds = train_encoder_results(dataset_name, loss_function, X_train, X_test, iterations, epochs, batch_size, input_dim, latent_dim, num_hidden, h_x, device)
        
        # Path for the CSV files
        train_losses_path = f'Results/TRAIN_{task}_{dataset_name}_{loss_function}_InputDim={input_dim}_LatentDim={latent_dim}_NumHidden={num_hidden}_TrainSize={train_size}_BatchSize={batch_size}_Epochs={epochs}_Iterations={iterations}.csv'
        test_losses_path = f'Results/TEST_{task}_{dataset_name}_{loss_function}_InputDim={input_dim}_LatentDim={latent_dim}_NumHidden={num_hidden}_TrainSize={train_size}_BatchSize={batch_size}_Epochs={epochs}_Iterations={iterations}.csv'
        lower_bounds_path = f'Results/LB_{task}_{dataset_name}_{loss_function}_InputDim={input_dim}_LatentDim={latent_dim}_NumHidden={num_hidden}_TrainSize={train_size}_BatchSize={batch_size}_Epochs={epochs}_Iterations={iterations}.csv'
        
        # Save matrices to CSV
        np.savetxt(train_losses_path, train_losses, delimiter=',')
        np.savetxt(test_losses_path, test_losses, delimiter=',')
        np.savetxt(lower_bounds_path, lower_bounds, delimiter=',')


    elif task == 'Regression':
        X_train, y_train, X_test, y_test, h_x = LoadDatasetReg(dataset_name, train_size, input_dim, output_dim)
        # Compare Autoencoder MSE With Lower Bound
        train_losses, test_losses, train_losses_reg, test_losses_reg, lower_bounds = train_regressor_results(dataset_name, loss_function, X_train, y_train, X_test, y_test, iterations, epochs, batch_size, input_dim, latent_dim, output_dim, num_hidden, h_x, device)
        
        # Path for the CSV files
        train_losses_path = f'Results/TRAIN_{task}_{dataset_name}_{loss_function}_InputDim={input_dim}_LatentDim={latent_dim}_NumHidden={num_hidden}_TrainSize={train_size}_BatchSize={batch_size}_Epochs={epochs}_Iterations={iterations}.csv'
        test_losses_path = f'Results/TEST_{task}_{dataset_name}_{loss_function}_InputDim={input_dim}_LatentDim={latent_dim}_NumHidden={num_hidden}_TrainSize={train_size}_BatchSize={batch_size}_Epochs={epochs}_Iterations={iterations}.csv'
        train_losses_reg_path = f'Results/TRAIN_REG_{task}_{dataset_name}_{loss_function}_InputDim={input_dim}_LatentDim={latent_dim}_NumHidden={num_hidden}_TrainSize={train_size}_BatchSize={batch_size}_Epochs={epochs}_Iterations={iterations}.csv'
        test_losses_reg_path = f'Results/TEST_REG_{task}_{dataset_name}_{loss_function}_InputDim={input_dim}_LatentDim={latent_dim}_NumHidden={num_hidden}_TrainSize={train_size}_BatchSize={batch_size}_Epochs={epochs}_Iterations={iterations}.csv'
        lower_bounds_path = f'Results/LB_{task}_{dataset_name}_{loss_function}_InputDim={input_dim}_LatentDim={latent_dim}_NumHidden={num_hidden}_TrainSize={train_size}_BatchSize={batch_size}_Epochs={epochs}_Iterations={iterations}.csv'
        
        # Save matrices to CSV
        np.savetxt(train_losses_path, train_losses, delimiter=',')
        np.savetxt(test_losses_path, test_losses, delimiter=',')
        np.savetxt(train_losses_reg_path, train_losses_reg, delimiter=',')
        np.savetxt(test_losses_reg_path, test_losses_reg, delimiter=',')
        np.savetxt(lower_bounds_path, lower_bounds, delimiter=',')