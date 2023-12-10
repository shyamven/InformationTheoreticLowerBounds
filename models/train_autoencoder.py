import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm
from utils.process_data import FindMSELowerBound
from models.classes import Autoencoder

# This function records the MSE loss across all epochs w.r.t the MSE lower bound
def train_autoencoder(X_train, X_test, i, iterations, epochs, batch_size, input_dim, latent_dim, num_hidden, MSE_LB, device='cpu'):
    # Initialize the autoencoder and optimizer
    autoencoder = Autoencoder(input_dim, latent_dim, num_hidden)   
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0005)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    
    # Instantiate train and test loss
    loss_train = np.zeros(epochs)
    loss_test = np.zeros(epochs)
    lower_bound = np.zeros(epochs)

    # Training the autoencoder
    for epoch in range(epochs):
        epoch_loss = 0
        
        for batch_X, in train_dataloader:
            autoencoder.train()
            
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = autoencoder(batch_X)
        
            error = torch.norm(batch_X - outputs, dim=1, p="fro")**2
            loss = torch.mean(error)
        
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        autoencoder.eval()
        
        # Validation data
        with torch.no_grad():
            outputs_test = autoencoder(X_test)
            test_error = torch.norm(X_test - outputs_test, dim=1, p="fro")**2
            test_loss = torch.mean(test_error)
            
        epoch_loss /= len(train_dataloader)
            
        loss_train[epoch] = epoch_loss
        loss_test[epoch] = test_loss.item()
        lower_bound[epoch] = MSE_LB

        print(f'Iteration [{i + 1}/{iterations}], Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, MSE Lower Bound: {MSE_LB:.4f}')

    return loss_train, loss_test, lower_bound


# A function that will return results accross multiple iterations
def train_encoder_results(DatasetName, X_train, X_test, iterations, epochs, batch_size, input_dim, latent_dim, num_hidden, device):
    # Initializing d, l, L, and K
    d = input_dim # Dimensionality of Input Space
    l = latent_dim # Dimensionality of Latent Space
    L = num_hidden + 1 # Number of Decoder Layers
    K = ((10**36)*(d**2) / 16)**L # Lipschitz Constant Upper Bound
    
    if DatasetName == 'Uniform':
        h = np.log(1) * d # Entropy for d (i.i.d) RVs: h(X) = np.log(1 - 0) * d
        
    elif DatasetName == 'TruncatedNormal':        
        # Parameters for the standard normal distribution truncated to [0,1]
        mu, sigma, a, b = 0, 1, 0, 1
        alpha = (a - mu) / sigma # Standardized lower bound
        beta = (b - mu) / sigma # Standardized upper bound
        Z = norm.cdf(beta) - norm.cdf(alpha) # Normalization constant Z
        
        # Entropy calculation using the provided formula
        entropy_trunc_normal = np.log(np.sqrt(2 * np.pi * np.e) * sigma * Z) + ((alpha * norm.pdf(alpha) - beta * norm.pdf(beta)) / (2 * Z))
        
        # Entropy for d (i.i.d) RVs
        h = d * entropy_trunc_normal
        
    elif DatasetName == 'ScaledUniform':
        h = np.log(0.5) * d # Entropy for d (i.i.d) RVs: h(X) = np.log(0.5 - 0) * d
        
    assert d >= l
    assert l > 0
    assert h <= 0
    assert K > 0

    # Calculating MSE Lower Bound
    MSE_LB = FindMSELowerBound(d, l, h, K)
    
    # Instantiate train and test loss across iterations
    train_losses = np.zeros((iterations, epochs))
    test_losses = np.zeros((iterations, epochs))
    lower_bounds = np.zeros((iterations, epochs))
    
    for i in range(iterations):
        loss_train, loss_test, lower_bound = train_autoencoder(X_train, X_test, i, iterations, epochs, batch_size, input_dim, latent_dim, num_hidden, MSE_LB, device=device)  
        train_losses[i,:] = loss_train
        test_losses[i,:] = loss_test
        lower_bounds[i,:] = lower_bound
        
    return train_losses, test_losses, lower_bounds
