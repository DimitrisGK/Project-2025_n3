#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import time
import numpy as np
import torch

# Fixed imports for modules/ folder
from modules.dataset_parser import DatasetParser
from modules.graph_utils import GraphBuilder
from modules.kahip_wrapper import partition_graph
from modules.models import MLPClassifier, ModelTrainer
from modules.index_builder import InvertedIndex, NeuralLSHIndex
from modules.metrics import MetricsCalculator



def parse_arguments():
    
    parser = argparse.ArgumentParser(
        description='Build Neural LSH index',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('-d', '--dataset', required=True,
                       help='Input dataset file')
    parser.add_argument('-i', '--index', required=True,
                       help='Output index path')
    parser.add_argument('-type', '--type', required=True, choices=['sift', 'mnist'],
                       help='Dataset type')
    
    # Graph construction
    parser.add_argument('--knn', type=int, default=10,
                       help='Number of neighbors for k-NN graph')
    
    # KaHIP parameters
    parser.add_argument('-m', '--partitions', type=int, default=100,
                       help='Number of partitions')
    parser.add_argument('--imbalance', type=float, default=0.03,
                       help='Imbalance parameter for KaHIP')
    parser.add_argument('--kahip_mode', type=int, default=2, choices=[0, 1, 2],
                       help='KaHIP mode: 0=FAST, 1=ECO, 2=STRONG')
    
    # MLP parameters
    parser.add_argument('--layers', type=int, default=3,
                       help='Number of MLP layers')
    parser.add_argument('--nodes', type=int, default=64,
                       help='Number of nodes per layer')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    # Data limit
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of points to load (for faster testing)')
    
    # Other
    parser.add_argument('--seed', type=int, default=1,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use for training')
    
    return parser.parse_args()

# Main execution function
def main():
    args = parse_arguments()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    print("="*70)
    print("NEURAL LSH INDEX BUILDER")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Type: {args.type}")
    print(f"Output: {args.index}")
    if args.limit:
        print(f"Limit: {args.limit} points")
    print(f"k-NN: {args.knn}")
    print(f"Partitions (m): {args.partitions}")
    print(f"MLP: {args.layers} layers × {args.nodes} nodes")
    print(f"Training: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Device: {args.device}")
    print("="*70)
    
    total_start = time.time()
    

    ## 1: Load Dataset

    print("\n[1/5] Loading dataset...")
    step_start = time.time()
    
    try:
        data = DatasetParser.load_dataset(args.dataset, args.type)
        
        # Apply limit if specified
        if args.limit and args.limit < len(data):
            print(f"  Limiting dataset from {len(data)} to {args.limit} points")
            data = data[:args.limit]
        
        print(f"Loaded {data.shape[0]} points of dimension {data.shape[1]}")
        print(f"Time: {time.time() - step_start:.2f}s")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        sys.exit(1)
    
    
    ## 2: Build k-NN Graph
        
    print("\n[2/5] Building k-NN graph...")
    step_start = time.time()
    
    try:
        graph_builder = GraphBuilder(data, k=args.knn, seed=args.seed)
        adjacency, vwgt, xadj, adjncy, adjcwgt = graph_builder.build_full_pipeline()
        print(f"Time: {time.time() - step_start:.2f}s")
    except Exception as e:
        print(f"Error building graph: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    
    ## 3: Graph Partitioning with KaHIP
  
    print("\n[3/5] Partitioning graph with KaHIP...")
    step_start = time.time()
    
    try:
        partition_labels, edge_cut = partition_graph(
            vwgt=vwgt,
            xadj=xadj,
            adjncy=adjncy,
            adjcwgt=adjcwgt,
            n_parts=args.partitions,
            imbalance=args.imbalance,
            mode=args.kahip_mode,
            seed=args.seed
        )
        print(f"Time: {time.time() - step_start:.2f}s")
    except Exception as e:
        print(f"Error in graph partitioning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    
    ## 4: Train MLP Classifier
    
    print("\n[4/5] Training MLP classifier...")
    step_start = time.time()
    
    try:
        # Create model
        model = MLPClassifier(
            input_dim=data.shape[1],
            hidden_dim=args.nodes,
            num_classes=args.partitions,
            num_layers=args.layers
        )
        
        # Train model
        trainer = ModelTrainer(model, device=args.device)
        history = trainer.train(
            data=data,
            labels=partition_labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        
        print(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%")
        print(f"Time: {time.time() - step_start:.2f}s")
    except Exception as e:
        print(f"Error training model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    
    ## 5: Build and Save Index
        
    print("\n[5/5] Building and saving index...")
    step_start = time.time()
    
    try:
        # Build inverted index
        inverted_index = InvertedIndex(partition_labels, data)
        inverted_index.print_stats()
        
        # Create metadata
        metadata = {
            'dataset_file': args.dataset,
            'dataset_type': args.type,
            'n_points': data.shape[0],
            'dimension': data.shape[1],
            'knn': args.knn,
            'partitions': args.partitions,
            'imbalance': args.imbalance,
            'kahip_mode': args.kahip_mode,
            'mlp_layers': args.layers,
            'mlp_nodes': args.nodes,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'seed': args.seed,
            'edge_cut': int(edge_cut),
            'final_val_acc': float(history['val_acc'][-1])
        }
        
        # Create and save index
        nlsh_index = NeuralLSHIndex(model, inverted_index, data, metadata)
        nlsh_index.save(args.index)
        
        print(f"Time: {time.time() - step_start:.2f}s")
    except Exception as e:
        print(f"Error saving index: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    
    # Summary
    
    total_time = time.time() - total_start
    
    print("\n" + "="*70)
    print("INDEX BUILDING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Index saved to: {args.index}")
    print(f"Partitions: {args.partitions}")
    print(f"Model validation accuracy: {history['val_acc'][-1]:.2f}%")
    print("="*70)


if __name__ == '__main__':
    main()