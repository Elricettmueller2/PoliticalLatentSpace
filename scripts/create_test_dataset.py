#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a small test dataset from full embeddings.
This script samples a subset of embeddings to create a smaller, GitHub-friendly dataset.
"""

import numpy as np
import h5py
import argparse
import os
import time
from tqdm import tqdm


def create_test_dataset(full_embedding_file, test_embedding_file, num_samples=10000, 
                       random_seed=42, preserve_words=None):
    """
    Create a small test dataset from the full embeddings.
    
    Args:
        full_embedding_file: Path to full HDF5 file containing embeddings
        test_embedding_file: Path to save the test dataset
        num_samples: Number of samples to include in test dataset
        random_seed: Random seed for reproducibility
        preserve_words: List of words that must be included in the test dataset
        
    Returns:
        Path to the created test dataset
    """
    print(f"Creating test dataset with {num_samples} samples...")
    start_time = time.time()
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Open full embedding file
    with h5py.File(full_embedding_file, 'r') as f_in:
        total = f_in['embeddings'].shape[0]
        dim = f_in['embeddings'].shape[1]
        
        print(f"Source file has {total} embeddings with dimension {dim}")
        
        # Get list of all words
        all_words = f_in['words'][:]
        
        # Convert bytes to strings if needed
        words_list = []
        for w in all_words:
            if isinstance(w, bytes):
                words_list.append(w.decode('utf-8'))
            else:
                words_list.append(w)
        
        # Find indices of words to preserve
        preserve_indices = []
        if preserve_words:
            print(f"Finding indices for {len(preserve_words)} words to preserve...")
            for word in preserve_words:
                if word in words_list:
                    idx = words_list.index(word)
                    preserve_indices.append(idx)
                else:
                    print(f"Warning: Word '{word}' not found in source file")
        
        # Sample random indices for the rest
        remaining_samples = num_samples - len(preserve_indices)
        if remaining_samples > 0:
            # Create mask to exclude already selected indices
            mask = np.ones(total, dtype=bool)
            mask[preserve_indices] = False
            
            # Get available indices
            available_indices = np.where(mask)[0]
            
            # Sample from available indices
            random_indices = np.random.choice(
                available_indices, 
                min(remaining_samples, len(available_indices)), 
                replace=False
            )
            
            # Combine preserved and random indices
            indices = np.concatenate([preserve_indices, random_indices])
        else:
            # If we have more preserved words than requested samples, take a subset
            indices = np.array(preserve_indices[:num_samples])
        
        # Sort indices for efficient reading
        indices.sort()
        
        print(f"Selected {len(indices)} embeddings for test dataset")
        
        # Create test dataset
        with h5py.File(test_embedding_file, 'w') as f_out:
            # Create datasets
            f_out.create_dataset('embeddings', shape=(len(indices), dim), dtype='float32')
            
            # Create words dataset with variable length strings
            dt = h5py.special_dtype(vlen=str)
            f_out.create_dataset('words', shape=(len(indices),), dtype=dt)
            
            # Copy data
            for i, idx in tqdm(enumerate(indices), total=len(indices), desc="Copying embeddings"):
                f_out['embeddings'][i] = f_in['embeddings'][idx]
                
                word = all_words[idx]
                if isinstance(word, bytes):
                    word = word.decode('utf-8')
                
                f_out['words'][i] = word
    
    end_time = time.time()
    print(f"Test dataset created in {end_time - start_time:.2f} seconds")
    print(f"Test dataset saved to {test_embedding_file}")
    
    return test_embedding_file


def main():
    parser = argparse.ArgumentParser(description='Create test dataset from full embeddings')
    parser.add_argument('--full-embedding-file', type=str, required=True, 
                        help='Path to full HDF5 file containing embeddings')
    parser.add_argument('--test-embedding-file', type=str, required=True, 
                        help='Path to save the test dataset')
    parser.add_argument('--num-samples', type=int, default=10000, 
                        help='Number of samples to include in test dataset')
    parser.add_argument('--random-seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--preserve-words', type=str, nargs='+', 
                        help='Words that must be included in the test dataset')
    args = parser.parse_args()
    
    create_test_dataset(
        args.full_embedding_file, 
        args.test_embedding_file, 
        args.num_samples,
        args.random_seed,
        args.preserve_words
    )


if __name__ == "__main__":
    main()
