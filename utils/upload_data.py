import torch
from scipy.stats import truncnorm, norm, beta as beta_dist
import numpy as np


def LoadDataset(DatasetName, train_size, input_dim):
    if DatasetName == "Uniform":
        # Use same dataset across different trials for reproducibility
        gen0 = torch.Generator()
        gen0 = gen0.manual_seed(0)
        X_train = torch.rand(train_size, input_dim, generator=gen0)
        X_test = torch.rand(10000, input_dim, generator=gen0)
        output1, output2 = X_train, X_test
        
        h_x = np.log(1) * input_dim # Entropy for d (i.i.d) RVs: h(X) = np.log(1 - 0) * d
        
    elif DatasetName == 'Beta':
        # Beta distribution parameters
        alpha_beta_dist = 100
        beta_beta_dist = 1
        
        # Generate data from the Beta distribution using NumPy
        np.random.seed(0) # For reproducibility
        X_train_np = np.random.beta(alpha_beta_dist, beta_beta_dist, (train_size, input_dim))
        X_test_np = np.random.beta(alpha_beta_dist, beta_beta_dist, (10000, input_dim))
        
        # Convert NumPy arrays to PyTorch tensors
        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        X_test = torch.tensor(X_test_np, dtype=torch.float32)
        output1, output2 = X_train, X_test
        
        h_x = beta_dist.entropy(alpha_beta_dist, beta_beta_dist) * input_dim # Entropy for d (i.i.d) RVs
        print(h_x)
        
    elif DatasetName == "TruncatedNormal":
        # Use same dataset across different trials for reproducibility
        gen0 = torch.Generator()
        gen0.manual_seed(0)
        X_train = generate_truncated_normal_data(train_size, input_dim)
        X_test = generate_truncated_normal_data(10000, input_dim)
        output1, output2 = X_train, X_test
        
        # Parameters for the standard normal distribution truncated to [0,1]
        mu, sigma, a, b = 0, 1, 0, 1
        alpha = (a - mu) / sigma # Standardized lower bound
        beta = (b - mu) / sigma # Standardized upper bound
        Z = norm.cdf(beta) - norm.cdf(alpha) # Normalization constant Z
        
        # Entropy calculation using the provided formula
        entropy_trunc_normal = np.log(np.sqrt(2 * np.pi * np.e) * sigma * Z) + ((alpha * norm.pdf(alpha) - beta * norm.pdf(beta)) / (2 * Z))
        
        h_x = entropy_trunc_normal * input_dim # Entropy for d (i.i.d) RVs
        
    elif DatasetName == 'ScaledUniform':
        # Use same dataset across different trials for reproducibility
        gen0 = torch.Generator()
        gen0 = gen0.manual_seed(0)
        X_train = torch.rand(train_size, input_dim, generator=gen0) * 0.5
        X_test = torch.rand(10000, input_dim, generator=gen0) * 0.5
        output1, output2 = X_train, X_test
        
        h_x = np.log(0.5) * input_dim # Entropy for d (i.i.d) RVs: h(X) = np.log(0.5 - 0) * d
        
    else:
        print("Error: Dataset name is undefined")
        exit(1)
  
    return output1, output2, h_x


def LoadDatasetReg(DatasetName, train_size, input_dim, output_dim):
    if DatasetName == "Uniform":
        # Use same dataset across different trials for reproducibility
        gen0 = torch.Generator()
        gen0 = gen0.manual_seed(0)
        X_train = torch.rand(train_size, input_dim, generator=gen0)
        X_test = torch.rand(10000, input_dim, generator=gen0)
        
        # Define the transformation weights and bias for y
        weights = torch.rand(input_dim, output_dim, generator=gen0)
        bias = torch.rand(output_dim, generator=gen0)
        
        # Generate y_train and y_test as a linear transformation of X with added noise
        y_train = X_train @ weights + bias + 0.1 * torch.randn(train_size, output_dim, generator=gen0)
        y_test = X_test @ weights + bias + 0.1 * torch.randn(10000, output_dim, generator=gen0)
        
        h_x = np.log(1) * input_dim # Entropy for d (i.i.d) RVs: h(X) = np.log(1 - 0) * d
        
        return X_train, y_train, X_test, y_test, h_x


def generate_truncated_normal_data(train_size, input_dim, lower_bound=0, upper_bound=1, mu=0, sigma=1):
    # Standardizing the bounds for truncation
    a, b = (lower_bound - mu) / sigma, (upper_bound - mu) / sigma

    # Generating truncated normal data
    data = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=(train_size, input_dim))

    # Converting to PyTorch tensor
    return torch.tensor(data, dtype=torch.float32)