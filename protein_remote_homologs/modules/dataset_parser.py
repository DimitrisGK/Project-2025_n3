#!/usr/bin/env python3

import struct
import numpy as np


class DatasetParser:
    """Parser for different dataset formats."""
    
    @staticmethod
    def load_dataset(filepath, dataset_type):
        """Args:
            filepath (str): Path to dataset file
            dataset_type (str): 'mnist' or 'sift'   
        Returns:
            np.ndarray: Dataset as (N, D) array"""
        if dataset_type == 'mnist':
            return DatasetParser._load_mnist(filepath)
        elif dataset_type == 'sift':
            return DatasetParser._load_sift(filepath)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")


    # Load MNIST dataset from .dat file
    # Format: Raw binary file with consecutive 32-bit floats (little-endian)
    # Each vector is 784 floats (28x28 image flattened)
    @staticmethod
    def _load_mnist(filepath):
        """Args:
            filepath (str): Path to .dat file
        Returns:
            np.ndarray: Dataset as (N, 784) array"""
        # Try to load as raw binary first
        try:
            data = np.fromfile(filepath, dtype=np.float32)
            dimension = 784
            
            # Check if data can be reshaped to (N, 784)
            if len(data) % dimension == 0:
                n_vectors = len(data) // dimension
                return data.reshape(n_vectors, dimension)
            else:
                raise ValueError(f"Data size {len(data)} is not divisible by {dimension}")
        
        except Exception as e:
            # Fallback: try to parse with struct (old method)
            print(f"  Warning: Using fallback parser ({e})")
            with open(filepath, 'rb') as f:
                file_data = f.read()
            
            dimension = 784
            bytes_per_vector = dimension * 4
            n_vectors = len(file_data) // bytes_per_vector
            
            vectors = []
            for i in range(n_vectors):
                offset = i * bytes_per_vector
                vector_bytes = file_data[offset:offset + bytes_per_vector]
                if len(vector_bytes) == bytes_per_vector:
                    vector = struct.unpack(f'{dimension}f', vector_bytes)
                    vectors.append(vector)
            
            return np.array(vectors, dtype=np.float32)
    

    # Load SIFT dataset from .fvecs file
    # Format: [dim(4 bytes)][vector(dim*4 bytes)][dim][vector]...
    # Each vector is prefixed by its dimension as int32
    @staticmethod
    def _load_sift(filepath):   # Load SIFT dataset from .fvecs file
        """Args:
            filepath (str): Path to .fvecs file
        Returns:
            np.ndarray: Dataset as (N, D) array"""
        vectors = []
        
        with open(filepath, 'rb') as f:
            while True:
                # Read dimension (4 bytes)
                dim_bytes = f.read(4)
                if not dim_bytes:
                    break
                
                dim = struct.unpack('i', dim_bytes)[0]
                
                # Read vector (dim * 4 bytes)
                vector_bytes = f.read(dim * 4)
                if len(vector_bytes) != dim * 4:
                    break
                
                vector = struct.unpack(f'{dim}f', vector_bytes)
                vectors.append(vector)
        
        return np.array(vectors, dtype=np.float32)
    

    # Load ground truth from .ivecs file (for SIFT).
    # Format: Same as .fvecs but with integers instead of floats
    @staticmethod
    def load_groundtruth(filepath):
        """Args:
            filepath (str): Path to .ivecs file
        Returns:
            np.ndarray: Ground truth as (N, K) array of indices"""
        
        vectors = []
        
        with open(filepath, 'rb') as f:
            while True:
                # Read dimension
                dim_bytes = f.read(4)
                if not dim_bytes:
                    break
                
                dim = struct.unpack('i', dim_bytes)[0]
                
                # Read vector of integers
                vector_bytes = f.read(dim * 4)
                if len(vector_bytes) != dim * 4:
                    break
                
                vector = struct.unpack(f'{dim}i', vector_bytes)
                vectors.append(vector)
        
        return np.array(vectors, dtype=np.int32)