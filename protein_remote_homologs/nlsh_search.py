#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import time
import numpy as np
import torch

# Imports for modules
from modules.dataset_parser import DatasetParser
from modules.graph_utils import GraphBuilder
from modules.kahip_wrapper import partition_graph
from modules.models import MLPClassifier, ModelTrainer
from modules.index_builder import InvertedIndex, NeuralLSHIndex
from modules.metrics import MetricsCalculator

# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Search using Neural LSH index',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('-d', '--dataset', required=True,
                       help='Dataset file')
    parser.add_argument('-q', '--query', required=True,
                       help='Query file')
    parser.add_argument('-i', '--index', required=True,
                       help='Index path')
    parser.add_argument('-o', '--output', required=True,
                       help='Output file')
    parser.add_argument('-type', '--type', required=True, choices=['sift', 'mnist'],
                       help='Dataset type')
    
    # Search parameters
    parser.add_argument('-N', '--neighbors', type=int, default=1,
                       help='Number of nearest neighbors')
    parser.add_argument('-R', '--radius', type=float, default=None,
                       help='Range search radius (default: 2000 for MNIST, 2800 for SIFT)')
    parser.add_argument('-T', '--probes', type=int, default=5,
                       help='Number of partitions to probe (multi-probe)')
    parser.add_argument('-range', '--range_search', type=str, default='true',
                       choices=['true', 'false'],
                       help='Perform range search')
    
    # Other
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    
    return parser.parse_args()

# Predict top-T partitions for a query
def predict_partitions(model, query, T, device='cpu'):
    """Args:
        model: Trained MLP classifier
        query (np.ndarray): Query vector
        T (int): Number of partitions to return
        device (str): Device
        
    Returns:
        np.ndarray: Top-T partition indices"""

    with torch.no_grad():
        query_tensor = torch.FloatTensor(query).unsqueeze(0).to(device)
        logits = model(query_tensor)
        probs = torch.softmax(logits, dim=1)
        
        # Get top-T partitions
        top_probs, top_indices = torch.topk(probs, k=min(T, probs.shape[1]), dim=1)
        
    return top_indices.cpu().numpy()[0]

# Perform exact k-NN search on candidates
def search_knn(query, candidates, data, N):
    """Args:
        query (np.ndarray): Query vector
        candidates (np.ndarray): Candidate point indices
        data (np.ndarray): Full dataset
        N (int): Number of neighbors
        
    Returns:
        tuple: (neighbor_indices, distances)"""

    if len(candidates) == 0:
        return np.array([]), np.array([])
    
    # Compute distances to all candidates
    candidate_data = data[candidates]
    distances = np.linalg.norm(candidate_data - query, axis=1)
    
    # Sort and take top-N
    sorted_idx = np.argsort(distances)[:N]
    neighbor_indices = candidates[sorted_idx]
    neighbor_distances = distances[sorted_idx]
    
    return neighbor_indices, neighbor_distances

# Perform range search on candidates
def search_range(query, candidates, data, radius):
    """Args:
        query (np.ndarray): Query vector
        candidates (np.ndarray): Candidate point indices
        data (np.ndarray): Full dataset
        radius (float): Search radius
        
    Returns:
        np.ndarray: Indices of neighbors within radius"""
    
    if len(candidates) == 0:
        return np.array([])
    
    # Compute distances to all candidates
    candidate_data = data[candidates]
    distances = np.linalg.norm(candidate_data - query, axis=1)
    
    # Filter by radius
    within_radius = candidates[distances <= radius]
    
    return within_radius

# Write results to output file in the required format
def write_output(output_file, results, metrics_summary):
    """Args:
        output_file (str): Output file path
        results (list): List of query results
        metrics_summary (dict): Summary metrics"""

    with open(output_file, 'w') as f:
        f.write("<METHOD NAME> [Neural LSH]\n")
        
        for result in results:
            f.write(f"Query: {result['query_id']}\n")   # Query ID (1-based)
            
            # k-NN results
            for i, (neighbor_id, approx_dist, true_dist) in enumerate(
                zip(result['neighbors'], result['approx_distances'], result['true_distances']), 1):
                # Neighbor ID (1-based)
                f.write(f"Nearest neighbor-{i}: {neighbor_id}\n")
                f.write(f"distanceApproximate: {approx_dist:.6f}\n")
                f.write(f"distanceTrue: {true_dist:.6f}\n")
            
            # Range search results
            if 'range_neighbors' in result and len(result['range_neighbors']) > 0:
                f.write("R-near neighbors:\n")
                for neighbor_id in result['range_neighbors']:
                    f.write(f"{neighbor_id}\n")
        
        # Summary metrics
        f.write(f"Average AF: {metrics_summary['average_af']:.6f}\n")
        f.write(f"Recall@N: {metrics_summary['recall_at_n']:.6f}\n")
        f.write(f"QPS: {metrics_summary['qps']:.6f}\n")
        f.write(f"tApproximateAverage: {metrics_summary['approx_time_avg']:.6f}\n")
        f.write(f"tTrueAverage: {metrics_summary['true_time_avg']:.6f}\n")


def main():
    
    args = parse_arguments()
    
    # Set default radius based on dataset type
    if args.radius is None:
        args.radius = 2000.0 if args.type == 'mnist' else 2800.0
    
    do_range_search = args.range_search.lower() == 'true'   # Parse range_search flag
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    print("="*70)
    print("NEURAL LSH SEARCH")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Queries: {args.query}")
    print(f"Index: {args.index}")
    print(f"Output: {args.output}")
    print(f"Type: {args.type}")
    print(f"N (neighbors): {args.neighbors}")
    print(f"R (radius): {args.radius}")
    print(f"T (probes): {args.probes}")
    print(f"Range search: {do_range_search}")
    print(f"Device: {args.device}")
    print("="*70)
    
    # Load Data
    print("\nLoading data...") 
    
    try:
        data = DatasetParser.load_dataset(args.dataset, args.type)
        queries = DatasetParser.load_dataset(args.query, args.type)
        print(f"✓ Loaded {data.shape[0]} data points, {queries.shape[0]} queries")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)
    
    # Load Index
    print("\nLoading index...") 
    
    try:
        nlsh_index = NeuralLSHIndex.load(args.index, device=args.device)
        model = nlsh_index.model
        inverted_index = nlsh_index.inverted_index
        
        if model is None or inverted_index is None:
            raise ValueError("Index is incomplete")
        
        print(f"✓ Index loaded successfully")
        print(f"  Partitions: {inverted_index.n_partitions}")
        print(f"  Model accuracy: {nlsh_index.metadata.get('final_val_acc', 'N/A')}")
    except Exception as e:
        print(f"✗ Error loading index: {e}")
        sys.exit(1)
    
    # Process Queries
    print(f"\nProcessing {len(queries)} queries...")
    
    metrics_calc = MetricsCalculator()
    results = []
    
    for query_idx, query in enumerate(queries):
        # Use 1-based IDs for output
        query_id = query_idx + 1
        
        
        # Approximate Search
        
        approx_start = time.time()
        
        # 1: Predict top-T partitions
        top_partitions = predict_partitions(model, query, args.probes, args.device)
        
        # 2: Collect candidates from these partitions
        candidates = inverted_index.get_candidates(top_partitions)
        
        # 3: Exact search on candidates (k-NN)
        neighbors, approx_distances = search_knn(query, candidates, data, args.neighbors)
        
        # 4: Range search (if enabled)
        range_neighbors = np.array([])
        if do_range_search:
            range_neighbors = search_range(query, candidates, data, args.radius)
        
        approx_time = time.time() - approx_start
        
        
        # Ground Truth (for metrics)
        
        eval_result = metrics_calc.evaluate_query_knn(
            query=query,
            approx_neighbors=neighbors,
            approx_distances=approx_distances,
            data=data,
            k=args.neighbors
        )
        
        # Add to metrics
        metrics_calc.add_knn_result(
            af=eval_result['approximation_factor'],
            recall=eval_result['recall'],
            approx_time=approx_time,
            true_time=eval_result['true_time']
        )
        
        # Convert indices to 1-based IDs for output
        neighbor_ids = neighbors + 1
        range_neighbor_ids = range_neighbors + 1 if len(range_neighbors) > 0 else []
        
        # Store result
        result = {
            'query_id': query_id,
            'neighbors': neighbor_ids,
            'approx_distances': approx_distances,
            'true_distances': eval_result['true_distances'][:len(neighbors)],
            'range_neighbors': range_neighbor_ids if do_range_search else []
        }
        results.append(result)
        
        # Progress
        if (query_idx + 1) % 10 == 0 or query_idx == len(queries) - 1:
            print(f"  Processed {query_idx + 1}/{len(queries)} queries")
    
    
    # Summary and Output
    
    print("\nComputing metrics...")
    metrics_summary = metrics_calc.get_summary()
    metrics_calc.print_summary()
    
    print(f"\nWriting results to {args.output}...")
    write_output(args.output, results, metrics_summary)
    
    print("\n" + "="*70)
    print("SEARCH COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Results written to: {args.output}")
    print(f"Average Approximation Factor: {metrics_summary['average_af']:.4f}")
    print(f"Recall@{args.neighbors}: {metrics_summary['recall_at_n']:.4f}")
    print(f"QPS: {metrics_summary['qps']:.2f}")
    print("="*70)


if __name__ == '__main__':
    main()