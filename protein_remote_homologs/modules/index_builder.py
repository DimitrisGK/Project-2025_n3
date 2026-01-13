#!/usr/bin/env python3
import os
import json
import numpy as np
import torch

# Inverted index mapping partitions to data points
class InvertedIndex:
  
    # Build inverted index
    def __init__(self, partition_labels, data):
        """Args:
            partition_labels (np.ndarray): Partition ID for each point
            data (np.ndarray): Original data (for storing IDs)"""
        
        self.n_partitions = int(partition_labels.max()) + 1
        self.partition_labels = partition_labels
        
        # Build inverted lists
        self.inverted_lists = {}
        for partition_id in range(self.n_partitions):
            # Get all point indices in this partition
            point_indices = np.where(partition_labels == partition_id)[0]
            self.inverted_lists[partition_id] = point_indices
        
        print(f"  ✓ Inverted index built: {self.n_partitions} partitions")
    
    # Get all candidate points from given partitions
    def get_candidates(self, partition_ids):
        """Args:
            partition_ids (np.ndarray or list): Partition IDs to probe
        Returns:
            np.ndarray: Indices of all points in these partitions"""

        candidates = []
        for pid in partition_ids:
            if pid in self.inverted_lists:
                candidates.extend(self.inverted_lists[pid])
        
        return np.array(candidates, dtype=np.int32)
    
    # Print statistics about the index
    def print_stats(self):
        sizes = [len(lst) for lst in self.inverted_lists.values()]
        print(f"  Partition statistics:")
        print(f"    Total partitions: {self.n_partitions}")
        print(f"    Min size: {min(sizes)}")
        print(f"    Max size: {max(sizes)}")
        print(f"    Avg size: {np.mean(sizes):.1f}")
        print(f"    Std size: {np.std(sizes):.1f}")


# Complete Neural LSH index
class NeuralLSHIndex:

    # Initialize Neural LSH index
    def __init__(self, model, inverted_index, data, metadata=None):
        """Args:
            model (nn.Module): Trained MLP classifier
            inverted_index (InvertedIndex): Inverted index
            data (np.ndarray): Original dataset
            metadata (dict): Metadata about the index"""

        self.model = model
        self.inverted_index = inverted_index
        self.data = data
        self.metadata = metadata or {}
    
    # Save index to disk
    def save(self, path):
        """Creates directory structure:
            path/
                model.pth          - PyTorch model
                inverted_index.npz - Inverted lists
                metadata.json      - Index metadata
                data.npy          - Original data
        
        Args:
            path (str): Directory path to save index"""
        
        os.makedirs(path, exist_ok=True)    # Create directory
        
        # Save model
        model_path = os.path.join(path, 'model.pth')
        torch.save(self.model.state_dict(), model_path)
        
        # Save inverted index
        index_path = os.path.join(path, 'inverted_index.npz')
        np.savez(
            index_path,
            n_partitions=self.inverted_index.n_partitions,
            partition_labels=self.inverted_index.partition_labels,
            **{f'partition_{i}': lst for i, lst in self.inverted_index.inverted_lists.items()}
        )
        
        # Save metadata
        metadata_path = os.path.join(path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save data
        data_path = os.path.join(path, 'data.npy')
        np.save(data_path, self.data)
        
        print(f"    Index saved to: {path}")
        print(f"    - {model_path}")
        print(f"    - {index_path}")
        print(f"    - {metadata_path}")
        print(f"    - {data_path}")
    
    @staticmethod
    # Load index from disk
    def load(path, device='cpu'):
        """Args:
            path (str): Directory path where index is saved
            device (str): Device to load model on
            
        Returns:
            NeuralLSHIndex: Loaded index"""

        # Load metadata
        metadata_path = os.path.join(path, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load data
        data_path = os.path.join(path, 'data.npy')
        data = np.load(data_path)
        
        # Load inverted index
        index_path = os.path.join(path, 'inverted_index.npz')
        index_data = np.load(index_path, allow_pickle=True)
        
        n_partitions = int(index_data['n_partitions'])
        partition_labels = index_data['partition_labels']
        
        # Reconstruct inverted lists
        inverted_lists = {}
        for i in range(n_partitions):
            key = f'partition_{i}'
            if key in index_data:
                inverted_lists[i] = index_data[key]
        
        # Create InvertedIndex object
        inverted_index = InvertedIndex.__new__(InvertedIndex)
        inverted_index.n_partitions = n_partitions
        inverted_index.partition_labels = partition_labels
        inverted_index.inverted_lists = inverted_lists
        
        # Load model
        from modules.models import MLPClassifier
        
        model = MLPClassifier(
            input_dim=metadata['dimension'],
            hidden_dim=metadata['mlp_nodes'],
            num_classes=metadata['partitions'],
            num_layers=metadata['mlp_layers']
        )
        
        model_path = os.path.join(path, 'model.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        return NeuralLSHIndex(model, inverted_index, data, metadata)