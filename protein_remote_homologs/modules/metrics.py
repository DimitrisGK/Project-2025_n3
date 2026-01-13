#!/usr/bin/env python3

import time
import numpy as np

# Calculate and track search metrics
class MetricsCalculator:
    
    # Initialize metrics calculator
    def __init__(self):
        self.results = []
    
    # Evaluate a single k-NN query
    def evaluate_query_knn(self, query, approx_neighbors, approx_distances, 
                           data, k):
        """Computes:
        - True k-NN (ground truth)
        - Approximation factor
        - Recall@k
        
        Args:
            query (np.ndarray): Query vector
            approx_neighbors (np.ndarray): Approximate neighbors (indices)
            approx_distances (np.ndarray): Approximate distances
            data (np.ndarray): Full dataset
            k (int): Number of neighbors
            
        Returns:
            dict: Evaluation results"""
        
        # Compute true k-NN (ground truth)
        true_start = time.time()
        true_distances = np.linalg.norm(data - query, axis=1)
        true_neighbors = np.argsort(true_distances)[:k]
        true_distances_sorted = true_distances[true_neighbors]
        true_time = time.time() - true_start
        
        # Compute approximation factor (AF)
        # AF = max(d_approx / d_true) for each returned neighbor
        if len(approx_neighbors) == 0:
            af = float('inf')
        else:
            # Get true distances for approximate neighbors
            true_dists_for_approx = true_distances[approx_neighbors]
            
            # AF = d_approx / d_true_kth
            # where d_true_kth is the distance to the k-th true neighbor
            d_true_kth = true_distances_sorted[-1] if len(true_distances_sorted) > 0 else 1.0
            
            if d_true_kth == 0:
                af = 1.0
            else:
                # Max ratio of approximate distances to k-th true distance
                af = np.max(approx_distances / d_true_kth) if len(approx_distances) > 0 else float('inf')
        
        # Compute Recall@k
        # Recall = |approx ∩ true| / k
        if len(approx_neighbors) == 0 or len(true_neighbors) == 0:
            recall = 0.0
        else:
            common = np.intersect1d(approx_neighbors, true_neighbors)
            recall = len(common) / k
        
        return {
            'approximation_factor': af,
            'recall': recall,
            'true_neighbors': true_neighbors,
            'true_distances': true_distances_sorted,
            'true_time': true_time
        }
    
    # Add k-NN search result to tracker
    def add_knn_result(self, af, recall, approx_time, true_time):
        """Args:
            af (float): Approximation factor
            recall (float): Recall@k
            approx_time (float): Approximate search time
            true_time (float): True search time"""

        self.results.append({
            'af': af,
            'recall': recall,
            'approx_time': approx_time,
            'true_time': true_time
        })
    
    # Get summary statistics
    def get_summary(self):
        """Returns:
            dict: Summary with average AF, Recall, QPS, times"""

        if not self.results:
            return {
                'average_af': 0.0,
                'recall_at_n': 0.0,
                'qps': 0.0,
                'approx_time_avg': 0.0,
                'true_time_avg': 0.0
            }
        
        # Filter out infinite AFs for averaging
        afs = [r['af'] for r in self.results if r['af'] != float('inf')]
        avg_af = np.mean(afs) if afs else float('inf')
        
        # Average recall
        recalls = [r['recall'] for r in self.results]
        avg_recall = np.mean(recalls)
        
        # Average times
        approx_times = [r['approx_time'] for r in self.results]
        true_times = [r['true_time'] for r in self.results]
        
        approx_time_avg = np.mean(approx_times)
        true_time_avg = np.mean(true_times)
        
        # QPS (Queries Per Second)
        qps = 1.0 / approx_time_avg if approx_time_avg > 0 else 0.0
        
        return {
            'average_af': avg_af,
            'recall_at_n': avg_recall,
            'qps': qps,
            'approx_time_avg': approx_time_avg,
            'true_time_avg': true_time_avg
        }
    
    # Print summary statistics
    def print_summary(self):
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("METRICS SUMMARY")
        print("="*70)
        print(f"Average Approximation Factor: {summary['average_af']:.6f}")
        print(f"Recall@N: {summary['recall_at_n']:.6f}")
        print(f"QPS (Queries Per Second): {summary['qps']:.2f}")
        print(f"Average Approximate Time: {summary['approx_time_avg']:.6f}s")
        print(f"Average True Time: {summary['true_time_avg']:.6f}s")
        print(f"Speedup: {summary['true_time_avg']/summary['approx_time_avg']:.2f}x")
        print("="*70)