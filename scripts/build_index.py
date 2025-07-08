#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build FAISS index for embeddings stored in HDF5 format.
This script creates an index file for fast similarity search.
"""

import numpy as np
import h5py
import faiss
import argparse
import os
import time
from tqdm import tqdm


def build_faiss_index(embedding_file, output_index=None, index_type="flat"):
    """
    Build a FAISS index for fast similarity search.
    
    Args:
        embedding_file: Path to HDF5 file containing embeddings
        output_index: Path to save the index (if None, will use same name as embedding file)
        index_type: Type of FAISS index to build (flat, ivf, hnsw)
    
    Returns:
        Path to the created index file
    """
    if output_index is None:
        output_index = os.path.splitext(embedding_file)[0] + '.index'
    
    print(f"Building index from {embedding_file}...")
    start_time = time.time()
    
    # Open embedding file
    with h5py.File(embedding_file, 'r') as f:
        # Get embedding dimension
        embeddings = f['embeddings']
        dim = embeddings.shape[1]
        total = embeddings.shape[0]
        
        print(f"Found {total} embeddings with dimension {dim}")
        
        # Create index based on type
        if index_type == "flat":
            # Simple flat index (exact search, but slower for large datasets)
            index = faiss.IndexFlatL2(dim)
        elif index_type == "ivf":
            # IVF index (approximate search, faster for large datasets)
            # We'll use 4*sqrt(n) centroids as a rule of thumb
            n_centroids = int(4 * np.sqrt(total))
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, n_centroids)
            # Need to train this index type
            print(f"Training IVF index with {n_centroids} centroids...")
            # Sample training vectors if there are too many
            if total > 1000000:
                train_size = 1000000
                train_indices = np.random.choice(total, train_size, replace=False)
                train_vectors = np.array([embeddings[i] for i in train_indices]).astype('float32')
            else:
                train_vectors = embeddings[:].astype('float32')
            index.train(train_vectors)
        elif index_type == "hnsw":
            # HNSW index (hierarchical navigable small world graphs)
            # Good balance of speed and accuracy
            index = faiss.IndexHNSWFlat(dim, 32)  # 32 neighbors per node
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add embeddings in chunks to avoid memory issues
        chunk_size = 10000
        for i in tqdm(range(0, total, chunk_size), desc="Adding to index"):
            end = min(i + chunk_size, total)
            chunk = embeddings[i:end].astype('float32')
            index.add(chunk)
    
    # Save index
    faiss.write_index(index, output_index)
    
    end_time = time.time()
    print(f"Index built in {end_time - start_time:.2f} seconds")
    print(f"Index saved to {output_index}")
    
    return output_index


def main():
    parser = argparse.ArgumentParser(description='Build FAISS index for embeddings')
    parser.add_argument('--embedding-file', type=str, required=True, 
                        help='Path to HDF5 file containing embeddings')
    parser.add_argument('--output-index', type=str, 
                        help='Path to save the index (default: same name as embedding file with .index extension)')
    parser.add_argument('--index-type', type=str, choices=['flat', 'ivf', 'hnsw'], default='flat',
                        help='Type of FAISS index to build')
    args = parser.parse_args()
    
    build_faiss_index(args.embedding_file, args.output_index, args.index_type)


if __name__ == "__main__":
    main()
