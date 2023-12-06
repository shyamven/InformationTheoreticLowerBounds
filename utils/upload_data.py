import torch
from utils.process_data import generate_truncated_normal_data

def LoadDataset(DatasetName, train_size, input_dim):
    if DatasetName == "Uniform":
        # Use same dataset across different trials for reproducibility
        gen0 = torch.Generator()
        gen0 = gen0.manual_seed(0)
        X_train = torch.rand(train_size, input_dim, generator=gen0)
        X_test = torch.rand(train_size, input_dim, generator=gen0)
        output1, output2 = X_train, X_test
        
    elif DatasetName == "TruncatedNormal":     
        gen0 = torch.Generator()
        gen0.manual_seed(0)
        X_train = generate_truncated_normal_data(train_size, input_dim)
        X_test = generate_truncated_normal_data(train_size, input_dim)
        output1, output2 = X_train, X_test
        
    else:
        print("Error: Dataset name is undefined")
        exit(1)
  
    return output1, output2