import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.process_data import ApproxMSELowerBound #, FindMSELowerBound
from models.classes import ClassicalNetwork, Decoder

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
def train_regressor(X_train, y_train, X_test, y_test, i, iterations, epochs, batch_size, input_dim, latent_dim, output_dim, num_hidden, MSE_LB, max_norm, device='cpu'):
    # Initialize the regression network and optimizer
    regressor = ClassicalNetwork(input_dim, latent_dim, output_dim, num_hidden)
    decoder = Decoder(input_dim, latent_dim, num_hidden)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(decoder.parameters(), lr=5e-3) # 5e-3
    optimizer_reg = optim.Adam(regressor.parameters(), lr=5e-3) # 5e-3
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)
    
    # Instantiate train and test loss
    loss_train = np.zeros(epochs)
    loss_train_reg = np.zeros(epochs)
    loss_test = np.zeros(epochs)
    loss_test_reg = np.zeros(epochs)
    lower_bound = np.zeros(epochs)

    for epoch in range(epochs):
        epoch_loss = 0; epoch_loss_reg = 0
        epoch_test_loss = 0; epoch_test_loss_reg = 0
        
        # Training the autoencoder
        for batch_X, batch_y in train_dataloader:
            decoder.train(); regressor.train()
            
            # Zero the gradients
            optimizer.zero_grad()
            optimizer_reg.zero_grad()
            
            # Forward pass
            preds, latents = regressor(batch_X)
            outputs = decoder(latents)
        
            # Backward pass
            loss = criterion(outputs, batch_X)
            loss_reg = criterion(preds, batch_y)
            loss.backward()
            loss_reg.backward()
            optimizer.step()
            optimizer_reg.step()
            
            # Enforce the max norm constraint
            for layer in decoder.children(): # children():
                if hasattr(layer, 'weight'):
                    enforce_max_norm(layer.weight, max_norm)
                    
            # Enforce the max norm constraint
            for layer in regressor.children(): # children():
                if hasattr(layer, 'weight'):
                    enforce_max_norm(layer.weight, max_norm)
            
        
        decoder.eval(); regressor.eval() 
        
        # Training data (computing error)
        with torch.no_grad():
            for batch_X, batch_y in train_dataloader:
                preds, latents = regressor(batch_X)
                outputs = decoder(latents)
                
                error = torch.mean(torch.norm(batch_X - outputs, dim=1)**2)
                error_reg = torch.mean(torch.norm(batch_y - preds, dim=1)**2)
                epoch_loss += error.item()
                epoch_loss_reg += error_reg.item()
        
        # Validation data (computing error)
        with torch.no_grad():
            for batch_X, batch_y in test_dataloader:
                preds_test, latents_test = regressor(batch_X)
                outputs_test = decoder(latents_test)
                
                test_error = torch.mean(torch.norm(batch_X - outputs_test, dim=1)**2)
                test_error_reg = torch.mean(torch.norm(batch_y - preds_test, dim=1)**2)
                epoch_test_loss += test_error.item()
                epoch_test_loss_reg += test_error_reg.item()
            
        epoch_loss /= len(train_dataloader)
        epoch_loss_reg /= len(train_dataloader)
        epoch_test_loss /= len(test_dataloader)
        epoch_test_loss_reg /= len(test_dataloader)
            
        loss_train[epoch] = epoch_loss
        loss_train_reg[epoch] = epoch_loss_reg
        loss_test[epoch] = epoch_test_loss
        loss_test_reg[epoch] = epoch_test_loss_reg
        lower_bound[epoch] = MSE_LB

        print(f'Iter [{i + 1}/{iterations}], Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Test Loss: {epoch_test_loss:.4f}, MSE LB: {MSE_LB:.4f}, Train Reg Loss: {epoch_loss_reg:.4f}, Test Reg Loss: {epoch_test_loss_reg:.4f}')

    return loss_train, loss_test, loss_train_reg, loss_test_reg, lower_bound


# A function that will return results accross multiple iterations
def train_regressor_results(dataset_name, X_train, y_train, X_test, y_test, iterations, epochs, batch_size, input_dim, latent_dim, output_dim, num_hidden, h_x, device):
    # Initializing d, l, L, and K
    d = input_dim # Dimensionality of Input Space
    l = latent_dim # Dimensionality of Latent Space
    L = num_hidden + 1 # Number of Decoder Layers
    max_norm = 2.25 # Max norm constraint on the weights
    # K = np.sqrt(((10**36)*(d**2) / 16)**L) # Lipschitz Constant Upper Bound
    K = (max_norm**(2*L)) / (16**L)
        
    assert l > 0
    assert h_x <= 0
    assert K > 0

    # Calculating MSE Lower Bound
    MSE_LB = ApproxMSELowerBound(d, l, h_x, K)
    
    # Instantiate train and test loss across iterations
    train_losses = np.zeros((iterations, epochs))
    train_losses_reg = np.zeros((iterations, epochs))
    test_losses = np.zeros((iterations, epochs))
    test_losses_reg = np.zeros((iterations, epochs))
    lower_bounds = np.zeros((iterations, epochs))
    
    for i in range(iterations):
        loss_train, loss_test, loss_train_reg, loss_test_reg, lower_bound = train_regressor(X_train, y_train, X_test, y_test, i, iterations, epochs, batch_size, input_dim, latent_dim, output_dim, num_hidden, MSE_LB, max_norm, device=device)  
        train_losses[i,:] = loss_train
        train_losses_reg[i,:] = loss_train_reg
        test_losses[i,:] = loss_test
        test_losses_reg[i,:] = loss_test_reg
        lower_bounds[i,:] = lower_bound
        
    return train_losses, test_losses, train_losses_reg, test_losses_reg, lower_bounds