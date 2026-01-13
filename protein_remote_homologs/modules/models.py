#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Multi-layer perceptron for partition label prediction
class MLPClassifier(nn.Module):
    
    # Initialize MLP classifier
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3):
        """Args:
            input_dim (int): Input dimension (D)
            hidden_dim (int): Hidden layer dimension
            num_classes (int): Number of output classes (m partitions)
            num_layers (int): Number of hidden layers"""
        
        super(MLPClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        # Build layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Dropout(0.3))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.3))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    # Forward pass
    def forward(self, x):
        """Args:
            x (torch.Tensor): Input batch of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)"""

        return self.network(x)
    
    # Predict class probabilities
    def predict_proba(self, x):
        """Args:
            x (torch.Tensor): Input batch
            
        Returns:
            torch.Tensor: Probabilities after softmax"""

        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs

# Dataset for training the partition classifier
class PartitionDataset(Dataset):
    
    # Initialize dataset
    def __init__(self, data, labels):
        """Args:
            data (np.ndarray): Data points of shape (N, D)
            labels (np.ndarray): Partition labels of shape (N,)"""

        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Trainer for the MLP classifier
class ModelTrainer:

    # Initialize trainer
    def __init__(self, model, device='cpu'):
        """Args:
            model (MLPClassifier): Model to train
            device (str): Device to use ('cpu' or 'cuda')"""

        self.model = model
        self.device = device
        self.model.to(device)
    
    # Train the model
    def train(self, data, labels, epochs=10, batch_size=128, lr=0.001, 
              validation_split=0.1):
        """Args:
            data (np.ndarray): Training data
            labels (np.ndarray): Partition labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            lr (float): Learning rate
            validation_split (float): Fraction of data for validation
            
        Returns:
            dict: Training history"""

        # Split data
        n_val = int(len(data) * validation_split)
        indices = np.random.permutation(len(data))
        
        train_data = data[indices[n_val:]]
        train_labels = labels[indices[n_val:]]
        val_data = data[indices[:n_val]]
        val_labels = labels[indices[:n_val]]
        
        # Create datasets
        train_dataset = PartitionDataset(train_data, train_labels)
        val_dataset = PartitionDataset(val_data, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=0)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        print(f"\nTraining MLP: {epochs} epochs, batch_size={batch_size}, lr={lr}")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward
                optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                # Backward
                loss.backward()
                optimizer.step()
                
                # Stats
                train_loss += loss.item() * batch_data.size(0)
                _, predicted = outputs.max(1)
                train_total += batch_labels.size(0)
                train_correct += predicted.eq(batch_labels).sum().item()
            
            train_loss /= train_total
            train_acc = 100.0 * train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data = batch_data.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item() * batch_data.size(0)
                    _, predicted = outputs.max(1)
                    val_total += batch_labels.size(0)
                    val_correct += predicted.eq(batch_labels).sum().item()
            
            val_loss /= val_total
            val_acc = 100.0 * val_correct / val_total
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
                  f"Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        
        print("✓ Training completed!")
        return history

# Test the MLP classifier
def test_mlp():
    print("Testing MLPClassifier...")
    
    # Synthetic data
    input_dim = 128
    num_classes = 50
    n_samples = 1000
    
    data = np.random.randn(n_samples, input_dim).astype(np.float32)
    labels = np.random.randint(0, num_classes, n_samples)
    
    # Create model
    model = MLPClassifier(input_dim, hidden_dim=64, num_classes=num_classes, 
                         num_layers=3)
    
    # Train
    trainer = ModelTrainer(model, device='cpu')
    history = trainer.train(data, labels, epochs=3, batch_size=64)
    
    # Test prediction
    test_input = torch.randn(10, input_dim)
    logits = model(test_input)
    probs = model.predict_proba(test_input)
    
    assert logits.shape == (10, num_classes)
    assert probs.shape == (10, num_classes)
    assert torch.allclose(probs.sum(dim=1), torch.ones(10), atol=1e-5)
    
    print("All tests passed!")


if __name__ == '__main__':
    test_mlp()