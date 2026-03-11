import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle

def load_adjacency_matrix(filepath):
    """
    Loads the pre-computed adjacency matrix from the .pkl file.
    """
    with open(filepath, 'rb') as f:
        # We add the latin1 fallback just in case it's an older Python 2 pickle file
        try:
            pickle_data = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            pickle_data = pickle.load(f, encoding='latin1')
            
    # Whether it's a list or tuple, the matrix is almost always the 3rd item (index 2)
    # The format is usually: [sensor_ids, sensor_id_to_ind, adj_mx]
    if isinstance(pickle_data, (tuple, list)) and len(pickle_data) >= 3:
        adj_mx = pickle_data[2] 
    else:
        adj_mx = pickle_data 
        
    # Convert to PyTorch tensor (wrapping in np.array first helps with weird pickle formats)
    return torch.FloatTensor(np.array(adj_mx))

def generate_dataset(data, seq_len, pre_len):
    """
    Chops the continuous time series into sliding windows.
    """
    X, Y = [], []
    # Slide a window across the data
    for i in range(len(data) - seq_len - pre_len + 1):
        X.append(data[i : i + seq_len])
        Y.append(data[i + seq_len : i + seq_len + pre_len])
    
    # Current X shape: (Batch, Time, Nodes)
    X = np.array(X) 
    Y = np.array(Y)
    
    # STGCN strictly expects X to be: (Batch, Channels, Nodes, Time)
    # We only have 1 channel (traffic volume)
    X = np.expand_dims(X, axis=1)        # Shape -> (Batch, 1, Time, Nodes)
    X = np.transpose(X, (0, 1, 3, 2))    # Shape -> (Batch, 1, Nodes, Time)
    
    # STGCN outputs Y as: (Batch, Nodes, Predicted_steps)
    Y = np.transpose(Y, (0, 2, 1))       # Shape -> (Batch, Nodes, Time)
    
    return X, Y

def get_dataloaders(csv_path, seq_len=12, pre_len=3, batch_size=32, train_ratio=0.7, val_ratio=0.1):
    """
    Main function to load the CSV, normalize it, and return PyTorch DataLoaders.
    """
    # 1. Load the raw volume data
    df = pd.read_csv(csv_path)
    data = df.values.astype(np.float32)
    
    # 2. Split into Train, Validation, and Test sets chronologically
    val_idx = int(len(data) * train_ratio)
    test_idx = int(len(data) * (train_ratio + val_ratio))
    
    train_data = data[:val_idx]
    val_data = data[val_idx:test_idx]
    test_data = data[test_idx:]
    
    # 3. Normalize the data (Z-Score)
    # We strictly compute mean/std ONLY on training data to prevent data leakage.
    mean = np.mean(train_data)
    std = np.std(train_data)
    
    train_data = (train_data - mean) / std
    val_data = (val_data - mean) / std
    test_data = (test_data - mean) / std
    
    # 4. Generate sliding windows
    X_train, y_train = generate_dataset(train_data, seq_len, pre_len)
    X_val, y_val = generate_dataset(val_data, seq_len, pre_len)
    X_test, y_test = generate_dataset(test_data, seq_len, pre_len)
    
    # 5. Convert to PyTorch Tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    # 6. Wrap in PyTorch DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    # Shuffle only the training data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, mean, std