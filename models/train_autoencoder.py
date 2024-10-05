import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.process_data import ApproxMSELowerBound #, FindMSELowerBound
from models.classes import Autoencoder

def enforce_max_norm(weight_matrix, max_norm):
    """
    Enforces the max norm constraint on the weight matrix.
    :param weight_matrix: The weight matrix of a neural network layer.
    :param max_norm: The maximum allowed Frobenius norm.
    """
    norm = weight_matrix.norm(p=2)
    if norm > max_norm:
        # Scale the weight matrix to enforce the max norm constraint
        weight_matrix.data = weight_matrix.data * max_norm / norm
        

# This function records the MSE loss across all epochs w.r.t the MSE lower bound
def train_autoencoder(loss_function, X_train, X_test, i, iterations, epochs, batch_size, input_dim, latent_dim, num_hidden, MSE_LB, max_norm, device='cpu'):
    # Initialize the autoencoder and optimizer
    autoencoder = Autoencoder(input_dim, latent_dim, num_hidden)
    criterion = torch.nn.MSELoss()
    if loss_function == 'MSE':
        optimizer = optim.Adam(autoencoder.parameters(), lr=5e-3) # 5e-3
    elif loss_function == 'MSEL2':
        optimizer = optim.Adam(autoencoder.parameters(), lr=5e-6, weight_decay=1e-5)
    else:
        print('Error: Loss function not recognized.')
        exit(1)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
    
    # Instantiate train and test loss
    loss_train = np.zeros(epochs)
    loss_test = np.zeros(epochs)
    lower_bound = np.zeros(epochs)

    # Training the autoencoder
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_test_loss = 0
        
        for _, batch_X in enumerate(train_dataloader):
            batch_X = batch_X[0]
            autoencoder.train()
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = autoencoder(batch_X)
            error = torch.mean(torch.norm(batch_X - outputs, dim=1)**2)
            epoch_loss += error.item()
        
            # Backward pass
            loss = criterion(batch_X, outputs)
            loss.backward()
            optimizer.step()
            
            # Enforce the max norm constraint
            for layer in autoencoder.decoder: # children():
                if hasattr(layer, 'weight'):
                    enforce_max_norm(layer.weight, max_norm)
            
        autoencoder.eval()
        
        # Validation data
        with torch.no_grad():
            for _, batch_X in enumerate(test_dataloader):
                batch_X = batch_X[0]
                outputs_test = autoencoder(batch_X)
                test_error = torch.mean(torch.norm(batch_X - outputs_test, dim=1)**2)
                epoch_test_loss += test_error.item()
            
        epoch_loss /= len(train_dataloader)
        epoch_test_loss /= len(test_dataloader)
            
        loss_train[epoch] = epoch_loss
        loss_test[epoch] = epoch_test_loss
        lower_bound[epoch] = MSE_LB

        print(f'Iteration [{i + 1}/{iterations}], Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Test Loss: {epoch_test_loss:.4f}, MSE Lower Bound: {MSE_LB:.4f}')

    return loss_train, loss_test, lower_bound


# A function that will return results accross multiple iterations
def train_encoder_results(dataset_name, loss_function, X_train, X_test, iterations, epochs, batch_size, input_dim, latent_dim, num_hidden, h_x, device):
    # Initializing d, l, L, and K
    d = input_dim # Dimensionality of Input Space
    l = latent_dim # Dimensionality of Latent Space
    L = num_hidden + 1 # Number of Decoder Layers
    max_norm = 10 # Max norm constraint on the weights
    # K = np.sqrt(((10**36)*(d**2) / 16)**L) # Lipschitz Constant Upper Bound
    K = (max_norm**(2*L)) / (16**L)
        
    assert l > 0
    assert h_x <= 0
    assert K > 0

    # Calculating MSE Lower Bound
    MSE_LB = ApproxMSELowerBound(d, l, h_x, K)
    
    # Instantiate train and test loss across iterations
    train_losses = np.zeros((iterations, epochs))
    test_losses = np.zeros((iterations, epochs))
    lower_bounds = np.zeros((iterations, epochs))
    
    for i in range(iterations):
        loss_train, loss_test, lower_bound = train_autoencoder(loss_function, X_train, X_test, i, iterations, epochs, batch_size, input_dim, latent_dim, num_hidden, MSE_LB, max_norm, device=device)  
        train_losses[i,:] = loss_train
        test_losses[i,:] = loss_test
        lower_bounds[i,:] = lower_bound
        
    return train_losses, test_losses, lower_bounds