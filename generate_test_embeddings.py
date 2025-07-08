import numpy as np
import h5py
import faiss
import os
from pathlib import Path
import random
import string

def generate_random_word(length=5):
    """Generate a random word of given length."""
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))

def generate_test_embeddings(num_words=10000, embedding_dim=384, output_dir='src/data/embeddings'):
    """
    Generate test word embeddings and save them to HDF5 file with FAISS index.
    
    Args:
        num_words: Number of words to generate
        embedding_dim: Dimension of embeddings
        output_dir: Directory to save files
    """
    print(f"Generating {num_words} test embeddings with dimension {embedding_dim}...")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate random words
    words = [generate_random_word(random.randint(3, 10)) for _ in range(num_words)]
    
    # Generate random embeddings
    embeddings = np.random.randn(num_words, embedding_dim).astype('float32')
    
    # Normalize embeddings (for cosine similarity)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    # Save to HDF5 file
    output_file = os.path.join(output_dir, 'test_embeddings.h5')
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('words', data=np.array(words, dtype='S10'))
        f.create_dataset('embeddings', data=embeddings)
    
    # Create FAISS index
    index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
    index.add(embeddings)
    
    # Save FAISS index
    index_file = os.path.join(output_dir, 'test_embeddings.index')
    faiss.write_index(index, index_file)
    
    print(f"Test embeddings saved to {output_file}")
    print(f"FAISS index saved to {index_file}")

if __name__ == "__main__":
    generate_test_embeddings()
