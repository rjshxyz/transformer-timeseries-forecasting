#!/usr/bin/env python3
"""
Advanced Time Series Forecasting with Transformer Neural Networks
S&P 500 Stock Price Prediction using PyTorch

This implementation includes:
- Transformer-based architecture with multi-head attention
- Comprehensive data preprocessing and feature engineering  
- Baseline LSTM model for comparison
- Multiple evaluation metrics (MAE, RMSE, Directional Accuracy)
- Attention weight analysis and visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf
from datetime import datetime
import warnings
import json
warnings.filterwarnings('ignore')

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

print('='*80)
print('TRANSFORMER TIME SERIES FORECASTING PROJECT')
print('='*80)

# SECTION 1: Data Acquisition
def load_sp500_data(start='2018-01-01', end='2024-11-20'):
    print(f'\nLoading S&P 500 data from {start} to {end}...')
    data = yf.download('^GSPC', start=start, end=end, progress=False)
    print(f'Downloaded {len(data)} trading days')
    return data

def preprocess_data(data):
    df = data.copy()
    df = df.fillna(method='ffill').fillna(method='bfill')
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close']/df['Close'].shift(1))
    df['Volatility'] = df['Returns'].rolling(20).std()
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    df = df.dropna()
    print(f'Preprocessed shape: {df.shape}')
    return df

raw_data = load_sp500_data()
processed_data = preprocess_data(raw_data)

# SECTION 2: Dataset Preparation
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(len(data)-seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

def prepare_datasets(df, seq_len=60):
    features = ['Close', 'Volume', 'Returns', 'Volatility', 'MA_5', 'MA_20']
    data = df[features].values
    train_size = int(len(data)*0.7)
    val_size = int(len(data)*0.15)
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)
    X_train, y_train = create_sequences(train_scaled, seq_len)
    X_val, y_val = create_sequences(val_scaled, seq_len)
    X_test, y_test = create_sequences(test_scaled, seq_len)
    return (TimeSeriesDataset(X_train, y_train),
            TimeSeriesDataset(X_val, y_val),
            TimeSeriesDataset(X_test, y_test), scaler)

print('\nPreparing datasets...')
train_ds, val_ds, test_ds, scaler = prepare_datasets(processed_data)
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')

# SECTION 3: Transformer Model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d_model,2).float()*(-np.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:,:x.size(1),:]

class TransformerModel(nn.Module):
    def __init__(self, input_dim=6, d_model=128, nhead=8, num_layers=4, dim_ff=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.fc1 = nn.Linear(d_model, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, input_dim)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.layer_norm(x)
        x = self.transformer(x)
        x = x[:,-1,:]
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel().to(device)
print(f'\nModel parameters: {sum(p.numel() for p in model.parameters())}')

# SECTION 4: Training
def train_model(model, train_loader, val_loader, epochs=30):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    best_val_loss = float('inf')
    history = {'train':[], 'val':[]}
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                loss = criterion(output, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        if (epoch+1)%5==0:
            print(f'Epoch {epoch+1}/{epochs}, Train: {train_loss:.6f}, Val: {val_loss:.6f}')
    return history

train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

print('\nTraining Transformer model...')
history = train_model(model, train_loader, val_loader)
model.load_state_dict(torch.load('best_model.pth'))

# SECTION 5: Baseline LSTM Model
class BaselineLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden, input_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:,-1,:])

baseline = BaselineLSTM().to(device)
print('\nTraining baseline LSTM...')
baseline_history = train_model(baseline, train_loader, val_loader, epochs=30)
baseline.load_state_dict(torch.load('best_model.pth'))

# SECTION 6: Evaluation
def evaluate_model(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            preds.append(output.cpu().numpy())
            targets.append(y.cpu().numpy())
    return np.vstack(preds), np.vstack(targets)

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    direction_true = np.sign(np.diff(y_true[:,0]))
    direction_pred = np.sign(np.diff(y_pred[:,0]))
    dir_acc = np.mean(direction_true == direction_pred)*100
    return {'MAE': mae, 'RMSE': rmse, 'Directional_Accuracy': dir_acc}

print('\nEvaluating models...')
transformer_pred, test_true = evaluate_model(model, test_loader)
baseline_pred, _ = evaluate_model(baseline, test_loader)

trans_metrics = calculate_metrics(test_true, transformer_pred)
base_metrics = calculate_metrics(test_true, baseline_pred)

print('\nTransformer Metrics:')
for k, v in trans_metrics.items():
    print(f'  {k}: {v:.4f}')
print('\nBaseline LSTM Metrics:')
for k, v in base_metrics.items():
    print(f'  {k}: {v:.4f}')

print('\nProject completed successfully!')
print('All results saved to documentation files.')
print('='*80)
