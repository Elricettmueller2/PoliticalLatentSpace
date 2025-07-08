import numpy as np
import h5py
import faiss
import os
from typing import List, Dict, Any, Optional, Union


class ChunkedEmbeddingStore:
    """
    Manages access to large word-level embeddings using chunked file storage and FAISS indexing.
    
    This class provides efficient access to a large embedding space by:
    1. Storing embeddings in HDF5 format for efficient disk access
    2. Using FAISS for fast similarity search
    3. Implementing caching for frequently accessed embeddings
    """
    
    def __init__(self, embedding_file: str, index_file: Optional[str] = None, 
                 cache_size: int = 1000, verbose: bool = False):
        """
        Initialize the chunked embedding store.
        
        Args:
            embedding_file: Path to HDF5 file containing embeddings
            index_file: Path to FAISS index file (if None, will look for default)
            cache_size: Number of embeddings to cache in memory
            verbose: Whether to print verbose output
        """
        self.embedding_file = embedding_file
        self.cache_size = cache_size
        self.verbose = verbose
        self.cache = {}  # Simple cache for frequently accessed embeddings
        
        # Use default index file if not provided
        if index_file is None:
            index_file = os.path.splitext(embedding_file)[0] + '.index'
        
        # Check if files exist
        if not os.path.exists(embedding_file):
            raise FileNotFoundError(f"Embedding file not found: {embedding_file}")
        
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        if self.verbose:
            print(f"Loading embeddings from {embedding_file}")
            print(f"Loading index from {index_file}")
        
        # Open embedding file
        self.h5_file = h5py.File(embedding_file, 'r')
        
        # Load word list
        self.words = self.h5_file['words'][:]
        
        # Get embedding dimension
        self.embedding_dim = self.h5_file['embeddings'].shape[1]
        
        # Load FAISS index
        self.index = faiss.read_index(index_file)
        
        if self.verbose:
            print(f"Loaded {len(self.words)} words with dimension {self.embedding_dim}")
    
    def get_nearest_words(self, query_embedding: Union[List[float], np.ndarray], 
                         top_n: int = 50, max_distance: float = 0.5) -> List[Dict[str, Any]]:
        """
        Find words closest to the query embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_n: Maximum number of words to return
            max_distance: Maximum distance threshold
            
        Returns:
            List of nearby words with distances and positions
        """
        # Convert query to numpy array of correct shape
        query_np = np.array(query_embedding).reshape(1, -1).astype('float32')
        
        # Search the index
        distances, indices = self.index.search(query_np, top_n)
        
        nearby_words = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if dist <= max_distance and idx < len(self.words):
                # Get word from index
                word = self.words[idx]
                if isinstance(word, bytes):
                    word = word.decode('utf-8')
                
                # Get embedding (from cache or file)
                if idx in self.cache:
                    word_embedding = self.cache[idx]
                else:
                    word_embedding = self.h5_file['embeddings'][idx]
                    # Update cache (simple LRU - remove oldest if full)
                    if len(self.cache) >= self.cache_size:
                        self.cache.pop(next(iter(self.cache)))
                    self.cache[idx] = word_embedding
                
                nearby_words.append({
                    'word': word,
                    'distance': float(dist),
                    'position': {
                        'embedding': word_embedding.tolist()
                    }
                })
                
        return nearby_words
    
    def get_word_embedding(self, word: str) -> Optional[np.ndarray]:
        """
        Get the embedding for a specific word.
        
        Args:
            word: Word to get embedding for
            
        Returns:
            Embedding vector or None if word not found
        """
        # Find word index
        for i, w in enumerate(self.words):
            if isinstance(w, bytes):
                w = w.decode('utf-8')
            if w == word:
                # Get embedding (from cache or file)
                if i in self.cache:
                    return self.cache[i]
                else:
                    embedding = self.h5_file['embeddings'][i]
                    # Update cache
                    if len(self.cache) >= self.cache_size:
                        self.cache.pop(next(iter(self.cache)))
                    self.cache[i] = embedding
                    return embedding
        
        return None
    
    def get_visualization_words(self, num_words: int = 100, 
                               dimension_scores: Optional[Dict[str, List[float]]] = None) -> Dict[str, Any]:
        """
        Get words for visualization.
        
        Args:
            num_words: Number of words to include
            dimension_scores: Dictionary mapping words to dimension scores
                             (if None, will sample random words)
            
        Returns:
            Dictionary with word data for visualization
        """
        if dimension_scores:
            # Use provided dimension scores
            words = list(dimension_scores.keys())[:num_words]
            x_values = [dimension_scores[w][0] for w in words]
            y_values = [dimension_scores[w][1] for w in words]
            sizes = [10 + abs(sum(dimension_scores[w])) * 5 for w in words]
            colors = ['rgba(0,0,255,0.7)' for _ in range(len(words))]
            scores = [sum(dimension_scores[w]) for w in words]
        else:
            # Sample random words
            indices = np.random.choice(len(self.words), min(num_words, len(self.words)), replace=False)
            words = []
            x_values = []
            y_values = []
            sizes = []
            colors = []
            scores = []
            
            for idx in indices:
                word = self.words[idx]
                if isinstance(word, bytes):
                    word = word.decode('utf-8')
                
                # Get embedding
                if idx in self.cache:
                    embedding = self.cache[idx]
                else:
                    embedding = self.h5_file['embeddings'][idx]
                
                # For visualization, we'll use the first two dimensions
                # In practice, you'd want to project to 2D using PCA, UMAP, etc.
                x_values.append(float(embedding[0]))
                y_values.append(float(embedding[1]))
                
                # Size based on vector magnitude
                size = 10 + np.linalg.norm(embedding) * 2
                sizes.append(float(size))
                
                # All same color for now
                colors.append('rgba(0,0,255,0.7)')
                
                # Score is just the magnitude
                scores.append(float(np.linalg.norm(embedding)))
                
                words.append(word)
        
        return {
            'words': words,
            'x': x_values,
            'y': y_values,
            'sizes': sizes,
            'colors': colors,
            'scores': scores
        }
    
    def project_to_reduced_space(self, word_indices: List[int], reducer) -> np.ndarray:
        """
        Project word embeddings to reduced space.
        
        Args:
            word_indices: Indices of words to project
            reducer: Dimensionality reduction model with transform method
            
        Returns:
            Projected coordinates
        """
        # Load embeddings for the specified indices
        embeddings = []
        for idx in word_indices:
            if idx in self.cache:
                embedding = self.cache[idx]
            else:
                embedding = self.h5_file['embeddings'][idx]
                # Update cache
                if len(self.cache) >= self.cache_size:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[idx] = embedding
            
            embeddings.append(embedding)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # Project to reduced space
        reduced = reducer.transform(embeddings_array)
        
        return reduced
    
    def close(self):
        """Close the HDF5 file."""
        self.h5_file.close()
        
    def __del__(self):
        """Destructor to ensure file is closed."""
        try:
            self.close()
        except:
            pass
