import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict
import math

class TradeDataset(Dataset):
    """Custom Dataset for trade time series data"""
    
    def __init__(self, data: pd.DataFrame, sequence_length: int = 6, 
                 target_cols: list = ['exports', 'imports'], 
                 scaler_X=None, scaler_y=None, country_encoder=None):
        """
        Args:
            data: DataFrame with columns [country, year, month, exports, imports]
            sequence_length: Number of past months to use for prediction
            target_cols: Columns to predict
            scaler_X, scaler_y: Pre-fitted scalers (for test set)
            country_encoder: Pre-fitted encoder (for test set)
        """
        self.sequence_length = sequence_length
        self.target_cols = target_cols
        
        # Encode countries
        if country_encoder is None:
            self.country_encoder = LabelEncoder()
            data['country_encoded'] = self.country_encoder.fit_transform(data['country'])
        else:
            self.country_encoder = country_encoder
            data['country_encoded'] = self.country_encoder.transform(data['country'])
        
        # Create cyclical time features
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        # Create year trend (normalized)
        data['year_trend'] = (data['year'] - data['year'].min()) / (data['year'].max() - data['year'].min())
        
        # Sort by country and date
        data = data.sort_values(['country', 'year', 'month']).reset_index(drop=True)
        
        # Create sequences per country
        self.sequences = []
        self.targets = []
        self.static_features = []
        
        for country in data['country'].unique():
            country_data = data[data['country'] == country].reset_index(drop=True)
            
            if len(country_data) < sequence_length + 1:
                continue
                
            for i in range(len(country_data) - sequence_length):
                # Sequence features
                seq = country_data.iloc[i:i+sequence_length]
                
                # Target (next time step)
                target = country_data.iloc[i+sequence_length]
                
                # Temporal features for sequence
                seq_features = seq[['exports', 'imports', 'month_sin', 'month_cos', 'year_trend']].values
                
                # Static features (country encoding, target time features)
                static = np.array([
                    target['country_encoded'],
                    target['month_sin'],
                    target['month_cos'],
                    target['year_trend']
                ])
                
                self.sequences.append(seq_features)
                self.targets.append(target[target_cols].values)
                self.static_features.append(static)
        
        # Convert to arrays
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        self.static_features = np.array(self.static_features, dtype=np.float32)
        
        # Apply log transformation to exports/imports (handle zeros)
        self.sequences[:, :, :2] = np.log1p(self.sequences[:, :, :2])
        self.targets = np.log1p(self.targets)
        
        # Fit or use scalers
        if scaler_X is None:
            self.scaler_X = StandardScaler()
            # Reshape for scaling
            n_samples, seq_len, n_features = self.sequences.shape
            self.sequences = self.scaler_X.fit_transform(
                self.sequences.reshape(-1, n_features)
            ).reshape(n_samples, seq_len, n_features)
        else:
            self.scaler_X = scaler_X
            n_samples, seq_len, n_features = self.sequences.shape
            self.sequences = self.scaler_X.transform(
                self.sequences.reshape(-1, n_features)
            ).reshape(n_samples, seq_len, n_features)
        
        if scaler_y is None:
            self.scaler_y = StandardScaler()
            self.targets = self.scaler_y.fit_transform(self.targets)
        else:
            self.scaler_y = scaler_y
            self.targets = self.scaler_y.transform(self.targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.static_features[idx]),
            torch.FloatTensor(self.targets[idx])
        )


class TradeForecaster(nn.Module):
    """Multi-task LSTM model for trade forecasting"""
    
    def __init__(self, 
                 seq_input_size: int = 5,  # exports, imports, month_sin, month_cos, year_trend
                 static_input_size: int = 4,  # country_encoded, month_sin, month_cos, year_trend
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 num_countries: int = 10,
                 embedding_dim: int = 16,
                 output_size: int = 2):  # exports, imports
        super(TradeForecaster, self).__init__()
        
        # Country embedding
        self.country_embedding = nn.Embedding(num_countries, embedding_dim)
        
        # LSTM for temporal sequence
        self.lstm = nn.LSTM(
            input_size=seq_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        combined_size = hidden_size + embedding_dim + static_input_size - 1  # -1 because country is embedded
        
        self.fc_layers = nn.Sequential(
            nn.Linear(combined_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )
        
    def forward(self, seq, static):
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(seq)
        lstm_features = lstm_out[:, -1, :]  # Last time step
        
        # Country embedding
        country_idx = static[:, 0].long()
        country_emb = self.country_embedding(country_idx)
        
        # Other static features (excluding country index)
        static_features = static[:, 1:]
        
        # Combine all features
        combined = torch.cat([lstm_features, country_emb, static_features], dim=1)
        
        # Final prediction
        output = self.fc_layers(combined)
        
        return output


def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader,
                num_epochs: int = 100,
                learning_rate: float = 0.001,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Train the forecasting model"""
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for seq, static, target in train_loader:
            seq, static, target = seq.to(device), static.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(seq, static)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq, static, target in val_loader:
                seq, static, target = seq.to(device), static.to(device), target.to(device)
                output = model(seq, static)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_trade_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return model, train_losses, val_losses


def predict(model: nn.Module, 
            data_loader: DataLoader,
            scaler_y: StandardScaler,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Make predictions and inverse transform"""
    
    model = model.to(device)
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for seq, static, target in data_loader:
            seq, static = seq.to(device), static.to(device)
            output = model(seq, static)
            predictions.append(output.cpu().numpy())
            actuals.append(target.numpy())
    
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    
    # Inverse transform
    predictions = scaler_y.inverse_transform(predictions)
    actuals = scaler_y.inverse_transform(actuals)
    
    # Inverse log transform
    predictions = np.expm1(predictions)
    actuals = np.expm1(actuals)
    
    return predictions, actuals


# Example usage
if __name__ == "__main__":
    # Load your data
    # df = pd.read_csv('trade_data.csv')
    # Expected columns: country, year, month, exports, imports
    
    # Example data structure:
    df = pd.DataFrame({
        'country': ['Afghanistan'] * 24 + ['Albania'] * 24,
        'year': [2015]*12 + [2016]*12 + [2015]*12 + [2016]*12,
        'month': list(range(1, 13)) * 4,
        'exports': np.random.randint(1000000, 100000000, 48),
        'imports': np.random.randint(1000000, 100000000, 48)
    })
    
    # Split data (temporal split)
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]
    
    # Create datasets
    train_dataset = TradeDataset(train_df, sequence_length=6)
    val_dataset = TradeDataset(val_df, sequence_length=6,
                               scaler_X=train_dataset.scaler_X,
                               scaler_y=train_dataset.scaler_y,
                               country_encoder=train_dataset.country_encoder)
    test_dataset = TradeDataset(test_df, sequence_length=6,
                                scaler_X=train_dataset.scaler_X,
                                scaler_y=train_dataset.scaler_y,
                                country_encoder=train_dataset.country_encoder)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    num_countries = len(df['country'].unique())
    model = TradeForecaster(
        seq_input_size=5,
        static_input_size=4,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        num_countries=num_countries,
        embedding_dim=16,
        output_size=2
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        num_epochs=100, learning_rate=0.001
    )
    
    # Make predictions
    predictions, actuals = predict(model, test_loader, train_dataset.scaler_y)
    
    # Calculate metrics
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-10))) * 100
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    
    print(f"\nTest Set Performance:")
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSE: {rmse:,.0f}")
    print(f"\nExports - MAPE: {np.mean(np.abs((actuals[:, 0] - predictions[:, 0]) / (actuals[:, 0] + 1e-10))) * 100:.2f}%")
    print(f"Imports - MAPE: {np.mean(np.abs((actuals[:, 1] - predictions[:, 1]) / (actuals[:, 1] + 1e-10))) * 100:.2f}%")