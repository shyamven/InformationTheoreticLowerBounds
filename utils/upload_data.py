import torch
from scipy.stats import truncnorm


def LoadDataset(DatasetName, train_size, input_dim):
    if DatasetName == "Uniform":
        # Use same dataset across different trials for reproducibility
        gen0 = torch.Generator()
        gen0 = gen0.manual_seed(0)
        X_train = torch.rand(train_size, input_dim, generator=gen0)
        X_test = torch.rand(train_size, input_dim, generator=gen0)
        output1, output2 = X_train, X_test
        
    elif DatasetName == "TruncatedNormal":
        # Use same dataset across different trials for reproducibility
        gen0 = torch.Generator()
        gen0.manual_seed(0)
        X_train = generate_truncated_normal_data(train_size, input_dim)
        X_test = generate_truncated_normal_data(train_size, input_dim)
        output1, output2 = X_train, X_test
        
    elif DatasetName == 'ScaledUniform':
        # Use same dataset across different trials for reproducibility
        gen0 = torch.Generator()
        gen0 = gen0.manual_seed(0)
        X_train = torch.rand(train_size, input_dim, generator=gen0) * 0.5
        X_test = torch.rand(train_size, input_dim, generator=gen0) * 0.5
        output1, output2 = X_train, X_test
        
    else:
        print("Error: Dataset name is undefined")
        exit(1)
  
    return output1, output2


def generate_truncated_normal_data(train_size, input_dim, lower_bound=0, upper_bound=1, mu=0, sigma=1):
    # Standardizing the bounds for truncation
    a, b = (lower_bound - mu) / sigma, (upper_bound - mu) / sigma

    # Generating truncated normal data
    data = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=(train_size, input_dim))

    # Converting to PyTorch tensor
    return torch.tensor(data, dtype=torch.float32)