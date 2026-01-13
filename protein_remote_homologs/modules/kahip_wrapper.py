#!/usr/bin/env python3

import numpy as np
import kahip


# Partition graph using KaHIP
def partition_graph(vwgt, xadj, adjncy, adjcwgt, n_parts=100, 
                    imbalance=0.03, mode=2, seed=1):
    """Args:
        vwgt (np.ndarray): Vertex weights (int32)
        xadj (np.ndarray): CSR index array (int32)
        adjncy (np.ndarray): CSR adjacency array (int32)
        adjcwgt (np.ndarray): Edge weights (int32)
        n_parts (int): Number of partitions
        imbalance (float): Imbalance parameter (0.03 = 3%)
        mode (int): KaHIP mode (0=FAST, 1=ECO, 2=STRONG)
        seed (int): Random seed
        
    Returns:
        tuple: (partition_labels, edge_cut)
            - partition_labels: array of partition IDs for each vertex
            - edge_cut: number of edges cut by the partitioning"""
    
    print(f"  Partitioning with KaHIP...")
    print(f"    Nodes: {len(vwgt)}")
    print(f"    Edges: {len(adjncy) // 2}")
    print(f"    Partitions: {n_parts}")
    print(f"    Imbalance: {imbalance}")
    print(f"    Mode: {['FAST', 'ECO', 'STRONG'][mode]}")
    
    # Number of vertices
    n_vertices = len(vwgt)
    
    # Convert imbalance to percentage (KaHIP expects 0-100)
    imbalance_percent = imbalance * 100
    
    # KaHIP kaffpa function signature:
    # kaffpa(vwgt, xadj, adjcwgt, adjncy, nparts, imbalance, suppress_output, seed, mode)

    # The order is:
    # 1. vwgt (vertex weights)
    # 2. xadj (CSR pointers)
    # 3. adjcwgt (edge weights)
    # 4. adjncy (CSR adjacency)
    # 5. nparts (number of partitions)
    # 6. imbalance (float)
    # 7. suppress_output (bool)
    # 8. seed (int)
    # 9. mode (int)
    
    try:
        edge_cut, partition_labels = kahip.kaffpa(
            vwgt.tolist(),            # vertex weights (list)
            xadj.tolist(),            # CSR index pointer (list)
            adjcwgt.tolist(),         # edge weights (list) - BEFORE adjncy!
            adjncy.tolist(),          # CSR adjacency (list) - AFTER adjcwgt!
            n_parts,                  # number of partitions (int)
            imbalance_percent,        # imbalance (float)
            True,                     # suppress output (bool)
            seed,                     # random seed (int)
            mode                      # mode (int: 0=FAST, 1=ECO, 2=STRONG)
        )
        
        partition_labels = np.array(partition_labels, dtype=np.int32)
        
        print(f"    Partitioning complete")
        print(f"    Edge cut: {edge_cut}")
        print(f"    Partition sizes:")
        
        # Print partition statistics
        unique, counts = np.unique(partition_labels, return_counts=True)
        min_size = counts.min()
        max_size = counts.max()
        avg_size = counts.mean()
        
        print(f"      Min: {min_size}, Max: {max_size}, Avg: {avg_size:.1f}")
        print(f"      Balance: {max_size/avg_size:.3f} (target: {1+imbalance:.3f})")
        
        return partition_labels, edge_cut
        
    except Exception as e:
        print(f"  ✗ KaHIP error: {e}")
        raise