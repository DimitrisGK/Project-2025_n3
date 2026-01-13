#!/usr/bin/env python3
import numpy as np

# Builds k-NN graph from dataset
class GraphBuilder:

    # Initialize GraphBuilder
    def __init__(self, data, k=10, seed=1):
        """Args:
            data (np.ndarray): Dataset (N, D)
            k (int): Number of neighbors
            seed (int): Random seed"""
        self.data = data
        self.k = k
        self.seed = seed
        self.n_points = data.shape[0]
        
        np.random.seed(seed)
    
    # Build k-NN graph - uses numpy (FAISS disabled due to macOS issues)
    def build_knn_graph(self):
        """Returns:
            dict: Graph with 'neighbors' (N, k) and 'distances' (N, k)"""
        # FAISS causes segfault on some macOS systems so, use numpy instead
        return self.build_knn_graph_numpy()
    
    # Build k-NN graph using FAISS (fast)
    def _build_knn_graph_faiss(self):
        import faiss
        print(f"  Building k-NN graph with k={self.k} (using FAISS)...")
        
        # Create FAISS index
        dimension = self.data.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        index.add(self.data)    # Add all points
        
        distances, neighbors = index.search(self.data, self.k + 1)  # Search for k+1 neighbors (including self)
        
        # Remove self (first neighbor is always the point itself)
        neighbors = neighbors[:, 1:]
        distances = distances[:, 1:]
        
        print(f"  ✓ k-NN graph built: {self.n_points} nodes, {self.n_points * self.k} edges")
        
        return {
            'neighbors': neighbors,
            'distances': distances
        }
    
    # Build k-NN graph using pure numpy (slow but works everywhere)
    def build_knn_graph_numpy(self):
        """Returns:
            dict: Graph with 'neighbors' (N, k) and 'distances' (N, k)"""
        
        print(f"  Building k-NN graph with k={self.k} (using numpy)...")
        if self.n_points > 5000:
            print(f"  WARNING: This may be slow for {self.n_points} points!")
        
        n = self.n_points
        neighbors = np.zeros((n, self.k), dtype=np.int32)
        distances = np.zeros((n, self.k), dtype=np.float32)
        
        batch_size = 100    # Process in batches to avoid memory issues
        
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            batch = self.data[i:end_i]
            
            # Compute distances from batch to all points
            # dist[b, j] = ||batch[b] - data[j]||^2
            diff = batch[:, np.newaxis, :] - self.data[np.newaxis, :, :]
            batch_distances = np.sum(diff ** 2, axis=2)
            
            # For each point in batch, find k+1 nearest (including self)
            for b in range(batch.shape[0]):
                point_idx = i + b
                dists = batch_distances[b]
                
                # Get k+1 smallest distances (k neighbors + self)
                k_plus_1 = min(self.k + 1, len(dists))
                nearest_indices = np.argpartition(dists, k_plus_1 - 1)[:k_plus_1]
                nearest_indices = nearest_indices[np.argsort(dists[nearest_indices])]
                
                nearest_indices = nearest_indices[1:][:self.k]  # Remove self (first one after sorting)
                
                # Pad if necessary
                if len(nearest_indices) < self.k:
                    padded = np.full(self.k, -1, dtype=np.int32)
                    padded[:len(nearest_indices)] = nearest_indices
                    neighbors[point_idx] = padded
                    
                    padded_dists = np.full(self.k, np.inf, dtype=np.float32)
                    padded_dists[:len(nearest_indices)] = np.sqrt(dists[nearest_indices])
                    distances[point_idx] = padded_dists
                else:
                    neighbors[point_idx] = nearest_indices
                    distances[point_idx] = np.sqrt(dists[nearest_indices])
            
            # Progress
            if (end_i) % 500 == 0 or end_i == n:
                print(f"    Progress: {end_i}/{n} points processed")
        
        print(f"  ✓ k-NN graph built: {self.n_points} nodes, {self.n_points * self.k} edges")
        
        return {
            'neighbors': neighbors,
            'distances': distances
        }
    
    # Convert directed k-NN graph to undirected graph with weights
    def symmetrize_graph(self, neighbors):
        """Weights:
        - 2 for mutual neighbors (edge in both directions)
        - 1 for one-way neighbors
        
        Args:
            neighbors (np.ndarray): k-NN neighbors (N, k)
        Returns:
            dict: Adjacency list with weights"""
        
        print("  Symmetrizing graph...")
        
        adjacency = {i: {} for i in range(self.n_points)}
        
        # Build adjacency with weights
        for i in range(self.n_points):
            for j in neighbors[i]:
                if j == -1:  # Invalid neighbor
                    continue
                
                # Check if edge already exists
                if j in adjacency[i]:
                    continue
                
                # Check if j also has i as neighbor (mutual)
                is_mutual = i in neighbors[j]
                weight = 2 if is_mutual else 1
                
                # Add edge (undirected, so add both directions)
                adjacency[i][j] = weight
                adjacency[j][i] = weight
        
        # Count edges
        n_edges = sum(len(adj) for adj in adjacency.values()) // 2
        print(f"  ✓ Graph symmetrized: {n_edges} undirected edges")
        
        return adjacency
    
    # Convert adjacency list to CSR format for KaHIP
    def to_csr_format(self, adjacency):
        """CSR Format:
        - vwgt: vertex weights (all 1s)
        - xadj: index array (starts of each vertex's neighbors)
        - adjncy: adjacency array (concatenated neighbor lists)
        - adjcwgt: edge weights
        
        Args:
            adjacency (dict): Adjacency list with weights
        Returns:
            tuple: (vwgt, xadj, adjncy, adjcwgt)"""
        
        print("  Converting to CSR format for KaHIP...")
        
        # Vertex weights (all nodes have weight 1)
        vwgt = np.ones(self.n_points, dtype=np.int32)
        
        # Build CSR arrays
        xadj = [0]
        adjncy = []
        adjcwgt = []
        
        for i in range(self.n_points):
            neighbors = sorted(adjacency[i].items())  # Sort for consistency
            
            for neighbor, weight in neighbors:
                adjncy.append(neighbor)
                adjcwgt.append(weight)
            
            xadj.append(len(adjncy))
        
        xadj = np.array(xadj, dtype=np.int32)
        adjncy = np.array(adjncy, dtype=np.int32)
        adjcwgt = np.array(adjcwgt, dtype=np.int32)
        
        print(f"  ✓ CSR format: {len(adjncy)} edges")
        
        return vwgt, xadj, adjncy, adjcwgt
    
    # Build complete k-NN graph pipeline
    def build_full_pipeline(self):
        """Returns:
            tuple: (adjacency, vwgt, xadj, adjncy, adjcwgt)"""
        
        # 1: Build k-NN graph
        knn_result = self.build_knn_graph()
        
        # 2: Symmetrize
        adjacency = self.symmetrize_graph(knn_result['neighbors'])
        
        # 3: Convert to CSR
        vwgt, xadj, adjncy, adjcwgt = self.to_csr_format(adjacency)
        
        return adjacency, vwgt, xadj, adjncy, adjcwgt