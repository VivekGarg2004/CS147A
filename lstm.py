import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

volume_df = pd.read_csv('data/sensor_volume_150.csv', header=0)
print(volume_df.shape)

adj_mat = pd.read_pickle('data/adj_mat_volume.pkl')

A = adj_mat[2].astype(np.float32)
np.fill_diagonal(A, 0)       
np.fill_diagonal(A, 1.0)       

degree = A.sum(axis=1, keepdims=True)
degree[degree == 0] = 1        
A_hat = A / degree           

print(A_hat.sum(axis=1)[:5])  
print(A_hat[0].max())         

A_hat_tensor = torch.FloatTensor(A_hat)

mean = volume_df.mean(axis=0)
std  = volume_df.std(axis=0)
std[std == 0] = 1  

volume_norm = (volume_df - mean) / std
data = volume_norm.values   

def create_sequences(data, input_len=12, output_len=3):
    """
    data: (T, N) numpy array
    Returns X: (samples, input_len, N), y: (samples, output_len, N)
    
    12 steps × 30 sec = 6 minutes of history → predict 3 steps (90 sec ahead)
    """
    X, y = [], []
    for i in range(len(data) - input_len - output_len + 1):
        X.append(data[i : i + input_len])          # (input_len, N)
        y.append(data[i + input_len : i + input_len + output_len])  # (output_len, N)
    return np.array(X), np.array(y)

INPUT_LEN  = 12   
OUTPUT_LEN = 3     

X, y = create_sequences(data, INPUT_LEN, OUTPUT_LEN)

n = len(X)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

class GraphConv(nn.Module):
    """GCN layer: X' = relu(A_hat @ X @ W + b)"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
                                
    def forward(self, x, A_hat):
        return torch.relu(self.linear(A_hat @ x))


class GCNLSTM(nn.Module):
    def __init__(self, num_nodes, in_features, gcn_hidden, lstm_hidden, output_len):
        super().__init__()
        self.gcn1 = GraphConv(in_features, gcn_hidden)
        self.gcn2 = GraphConv(gcn_hidden, gcn_hidden)
        self.dropout = nn.Dropout(0.3)

        self.lstm = nn.LSTM(gcn_hidden, lstm_hidden, batch_first=True,
                            num_layers=2, dropout=0.2)

        self.output = nn.Linear(lstm_hidden, output_len)
        self.output_len = output_len
        self.num_nodes = num_nodes

    def forward(self, x, A_hat):
        batch, T, N = x.shape

        gcn_out = []
        for t in range(T):
            xt = x[:, t, :].unsqueeze(-1)    
            ht = self.gcn1(xt, A_hat)          
            ht = self.dropout(ht)
            ht = self.gcn2(ht, A_hat)           
            gcn_out.append(ht)

        gcn_seq = torch.stack(gcn_out, dim=1)


        gcn_seq = gcn_seq.permute(0, 2, 1, 3)          
        gcn_seq = gcn_seq.reshape(batch * N, T, -1)        

        lstm_out, _ = self.lstm(gcn_seq)                   
        last = lstm_out[:, -1, :]                          

        out = self.output(last)                        
        out = out.reshape(batch, N, self.output_len)       
        return out.permute(0, 2, 1)                        
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
A_hat_tensor = A_hat_tensor.to(device)

def to_loader(X_arr, y_arr, batch_size=64, shuffle=True):
    ds = TensorDataset(torch.FloatTensor(X_arr), torch.FloatTensor(y_arr))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = to_loader(X_train, y_train, shuffle=True)
val_loader   = to_loader(X_val, y_val, shuffle=False)

model = GCNLSTM(
    num_nodes=150, in_features=1, gcn_hidden=32,
    lstm_hidden=64, output_len=OUTPUT_LEN
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
criterion = nn.MSELoss()

EPOCHS = 100
best_val_loss = float('inf')
patience_counter = 0
EARLY_STOP_PATIENCE = 15

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb, A_hat_tensor)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_loss += criterion(model(xb, A_hat_tensor), yb).item()

    avg_val = val_loss / len(val_loader)
    scheduler.step(avg_val)

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        patience_counter = 0
        best_state = model.state_dict().copy()
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0 or patience_counter == 0:
        print(f"Epoch {epoch+1}: train={train_loss/len(train_loader):.4f}  "
              f"val={avg_val:.4f}  best={best_val_loss:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.1e}")

    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(best_state)
print(f"Best model (val loss: {best_val_loss:.4f})")

mean_t = torch.FloatTensor(mean.values).to(device)
std_t  = torch.FloatTensor(std.values).to(device)

model.eval()
preds, trues = [], []
with torch.no_grad():
    for xb, yb in to_loader(X_test, y_test, shuffle=False):
        xb = xb.to(device)
        pred = model(xb, A_hat_tensor)  
        # Denormalize
        pred = pred * std_t + mean_t
        yb   = yb.to(device) * std_t + mean_t
        preds.append(pred.cpu().numpy())
        trues.append(yb.cpu().numpy())

preds = np.concatenate(preds, axis=0)   
trues = np.concatenate(trues, axis=0)

#Metrics
mae  = np.mean(np.abs(preds - trues))
rmse = np.sqrt(np.mean((preds - trues)**2))

mask = trues > 10.0
mape = np.mean(np.abs((preds[mask] - trues[mask]) / trues[mask])) * 100

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"MAE/Mean: {mae / volume_df.values.mean() * 100:.1f}%")  # relative error

SAVE_PATH = 'gcn_lstm_checkpoint.pth'

checkpoint = {
    'model_state_dict': model.state_dict(),
    'hyperparameters': {
        'num_nodes': 150,
        'in_features': 1,
        'gcn_hidden': 32,
        'lstm_hidden': 64,
        'output_len': OUTPUT_LEN
    }, 
#To normalize for future inference 
    'scaler_mean': mean.values, 
    'scaler_std': std.values
}

torch.save(checkpoint, SAVE_PATH)
print(f"Model checkpoint successfully saved to {SAVE_PATH}")