import torch
from torchvision import datasets, transforms
from scipy.stats import gaussian_kde
from scipy.stats import truncnorm, norm #, beta as beta_dist
import numpy as np

# Loading Datasets for Autoencoder Task
def LoadDataset(DatasetName, train_size, input_dim):
    if DatasetName == "Uniform":
        # Use same dataset across different trials for reproducibility
        gen0 = torch.Generator()
        gen0 = gen0.manual_seed(0)
        X_train = torch.rand(train_size, input_dim, generator=gen0)
        X_test = torch.rand(10000, input_dim, generator=gen0)
        
        h_x = np.log(1) * input_dim # Entropy for d (i.i.d) RVs: h(X) = np.log(1 - 0) * d
        
        print(f'Dataset Differential Entropy: h(x) = {h_x}')
        return X_train, X_test, h_x
        
    elif DatasetName == "TruncatedNormal":
        # Use same dataset across different trials for reproducibility
        gen0 = torch.Generator()
        gen0.manual_seed(0)
        X_train = generate_truncated_normal_data(train_size, input_dim)
        X_test = generate_truncated_normal_data(10000, input_dim)
        
        # Parameters for the standard normal distribution truncated to [0,1]
        mu, sigma, a, b = 0, 1, 0, 1
        alpha = (a - mu) / sigma # Standardized lower bound
        beta = (b - mu) / sigma # Standardized upper bound
        Z = norm.cdf(beta) - norm.cdf(alpha) # Normalization constant Z
        
        # Entropy calculation using the provided formula
        entropy_trunc_normal = np.log(np.sqrt(2 * np.pi * np.e) * sigma * Z) + ((alpha * norm.pdf(alpha) - beta * norm.pdf(beta)) / (2 * Z))
        
        h_x = entropy_trunc_normal * input_dim # Entropy for d (i.i.d) RVs
        
        print(f'Dataset Differential Entropy: h(x) = {h_x}')
        return X_train, X_test, h_x
        
    elif DatasetName == 'MNIST':
        transform = transforms.ToTensor()
        mnist_train = datasets.MNIST(root='data/', train=True, transform=transform, download=True)
        mnist_test = datasets.MNIST(root='data/', train=False, transform=transform, download=True)
        
        # Convert training and test sets into tensors
        X_train = torch.stack([img.flatten() for img, _ in mnist_train])  # Shape: [60000, 784]
        X_test = torch.stack([img.flatten() for img, _ in mnist_test])    # Shape: [10000, 784]
        np.random.seed(0); indices = np.random.choice(np.arange(len(mnist_train)), train_size, replace=False)
        X_train = X_train[indices]
                
        h_x, mse_degen, degen_dim = EstimateMNISTEntropy(X_test)
        
        print(f'Dataset Differential Entropy: h(x) = {h_x}')
        return X_train, X_test, h_x, mse_degen, degen_dim
        
    # elif DatasetName == 'Beta':
    #     # Beta distribution parameters
    #     alpha_beta_dist = 100
    #     beta_beta_dist = 1
        
    #     # Generate data from the Beta distribution using NumPy
    #     np.random.seed(0) # For reproducibility
    #     X_train_np = np.random.beta(alpha_beta_dist, beta_beta_dist, (train_size, input_dim))
    #     X_test_np = np.random.beta(alpha_beta_dist, beta_beta_dist, (10000, input_dim))
        
    #     # Convert NumPy arrays to PyTorch tensors
    #     X_train = torch.tensor(X_train_np, dtype=torch.float32)
    #     X_test = torch.tensor(X_test_np, dtype=torch.float32)
        
    #     h_x = beta_dist.entropy(alpha_beta_dist, beta_beta_dist) * input_dim # Entropy for d (i.i.d) RVs
    #     return X_train, X_test, h_x
        
    # elif DatasetName == 'ScaledUniform':
    #     # Use same dataset across different trials for reproducibility
    #     gen0 = torch.Generator()
    #     gen0 = gen0.manual_seed(0)
    #     X_train = torch.rand(train_size, input_dim, generator=gen0) * 0.5
    #     X_test = torch.rand(10000, input_dim, generator=gen0) * 0.5
        
    #     h_x = np.log(0.5) * input_dim # Entropy for d (i.i.d) RVs: h(X) = np.log(0.5 - 0) * d
    #     return X_train, X_test, h_x
        
    else:
        print("Error: Dataset name is undefined")
        exit(1)


# Loading Datasets for Regression Task
def LoadDatasetReg(DatasetName, train_size, input_dim, output_dim):
    if DatasetName == "Uniform":
        # Use same dataset across different trials for reproducibility
        gen0 = torch.Generator()
        gen0 = gen0.manual_seed(0)
        
        X_train = torch.rand(train_size, input_dim, generator=gen0)
        X_test = torch.rand(10000, input_dim, generator=gen0)
        
        h_x = np.log(1) * input_dim # Entropy for d (i.i.d) RVs: h(X) = np.log(1 - 0) * d
    
    elif DatasetName == "TruncatedNormal":
        # Use same dataset across different trials for reproducibility
        gen0 = torch.Generator()
        gen0.manual_seed(0)
        X_train = generate_truncated_normal_data(train_size, input_dim)
        X_test = generate_truncated_normal_data(10000, input_dim)
        
        # Parameters for the standard normal distribution truncated to [0,1]
        mu, sigma, a, b = 0, 1, 0, 1
        alpha = (a - mu) / sigma # Standardized lower bound
        beta = (b - mu) / sigma # Standardized upper bound
        Z = norm.cdf(beta) - norm.cdf(alpha) # Normalization constant Z
        
        # Entropy calculation using the provided formula
        entropy_trunc_normal = np.log(np.sqrt(2 * np.pi * np.e) * sigma * Z) + ((alpha * norm.pdf(alpha) - beta * norm.pdf(beta)) / (2 * Z))
        
        h_x = entropy_trunc_normal * input_dim # Entropy for d (i.i.d) RVs
        
    else:
        print("Error: Dataset name is undefined")
        exit(1)
    
    # Define the transformation weights and bias for y
    weights = torch.rand(input_dim, output_dim, generator=gen0)
    bias = torch.rand(output_dim, generator=gen0)
    
    # Generate y_train and y_test as a linear transformation of X with added noise
    y_train = X_train @ weights + bias + 0.1*torch.randn(train_size, output_dim, generator=gen0)
    y_test = X_test @ weights + bias + 0.1*torch.randn(10000, output_dim, generator=gen0)
    
    print(f'Dataset Differential Entropy: h(x) = {h_x}')
    return X_train, y_train, X_test, y_test, h_x


# Loading Datasets for Classification Task
def LoadDatasetClass(DatasetName, train_size, input_dim, output_dim):
    if DatasetName == 'MNIST':
        transform = transforms.ToTensor()
        mnist_train = datasets.MNIST(root='data/', train=True, transform=transform, download=True)
        mnist_test = datasets.MNIST(root='data/', train=False, transform=transform, download=True)
        
        # Convert training and test sets into tensors
        X_train = torch.stack([img.flatten() for img, _ in mnist_train])  # Shape: [60000, 784]
        y_train = torch.tensor([label for _, label in mnist_train], dtype=torch.long)  # Shape: [60000]
        X_test = torch.stack([img.flatten() for img, _ in mnist_test])    # Shape: [10000, 784]
        y_test = torch.tensor([label for _, label in mnist_test], dtype=torch.long) # Shape: [10000]
        np.random.seed(0); indices = np.random.choice(np.arange(len(mnist_train)), train_size, replace=False)
        X_train = X_train[indices]; y_train = y_train[indices]
        
        h_x, mse_degen, degen_dim = EstimateMNISTEntropy(X_test)
        
    else:
        print("Error: Dataset name is undefined")
        exit(1)
    
    print(f'Dataset Differential Entropy: h(x) = {h_x}')
    return X_train, y_train, X_test, y_test, h_x, mse_degen, degen_dim
    

def generate_truncated_normal_data(train_size, input_dim, lower_bound=0, upper_bound=1, mu=0, sigma=1):
    # Standardizing the bounds for truncation
    a, b = (lower_bound - mu) / sigma, (upper_bound - mu) / sigma

    # Generating truncated normal data
    data = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=(train_size, input_dim))

    # Converting to PyTorch tensor
    return torch.tensor(data, dtype=torch.float32)


def EstimateMNISTEntropy(X_test):
    # Compute the variance of each dimension across all samples
    X_test = X_test.numpy()
    variances = np.var(X_test, axis=0)  # Shape: [input_dim]
    non_zero_variance_mask = variances > 1e-2  # Create a mask for dimensions with non-zero variance
    X_test = X_test[:, non_zero_variance_mask]  # Apply the mask to X_train and X_test
    print(f"Number of nondegenerate dimensions: {X_test.shape[1]}")
    
    # Compute MSE of the "degenerate distribution" cases
    mse_degen = np.sum(variances[~non_zero_variance_mask])
    degen_dim = np.sum(~non_zero_variance_mask)
    
    n, d = X_test.shape  # n = 10000, d = input_dim - degen_dim
    
    # Step 2: Kernel Density Estimation using gaussian_kde from scipy
    kde = gaussian_kde(X_test.T)  # Note: gaussian_kde expects transposed data
    
    # Step 3: Compute KDE for all training points
    p_hat_train = kde(X_test.T)  # Returns the density estimates for the training points
    
    # Compute the log density estimates
    log_p_hat_train = np.log(p_hat_train + 1e-10)  # Add small constant for numerical stability
    
    # Step 4: Compute the differential entropy estimate
    h_hat_train = -np.mean(log_p_hat_train)
    
    return h_hat_train, mse_degen, degen_dim