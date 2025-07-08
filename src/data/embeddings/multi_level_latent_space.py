import numpy as np
import h5py
import faiss
import json
import os
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

from .chunked_embedding_store import ChunkedEmbeddingStore


class MultiLevelLatentSpace:
    """
    Manages a multi-level latent space that combines:
    1. Entity-level embeddings (parties, politicians) - compact, interpretable
    2. Word-level embeddings - detailed, rich
    
    This hybrid approach enables:
    - High-level political landscape visualization (entity-level)
    - Detailed semantic exploration (word-level)
    - Drill-down from entities to their most relevant words and concepts
    """
    
    def __init__(self, 
                 entity_data_path: Optional[str] = None,
                 word_embedding_path: Optional[str] = None,
                 index_file: Optional[str] = None,
                 cache_size: int = 1000,
                 verbose: bool = False):
        """
        Initialize the multi-level latent space.
        
        Args:
            entity_data_path: Path to JSON file containing entity-level data
            word_embedding_path: Path to HDF5 file containing word embeddings
            index_file: Path to FAISS index file (if None, will look for default)
            cache_size: Number of embeddings to cache in memory
            verbose: Whether to print verbose output
        """
        self.verbose = verbose
        self.entity_data = {}
        self.word_embedding_store = None
        
        # Load entity-level data if provided
        if entity_data_path and os.path.exists(entity_data_path):
            if self.verbose:
                print(f"Loading entity data from {entity_data_path}")
            try:
                with open(entity_data_path, 'r', encoding='utf-8') as f:
                    self.entity_data = json.load(f)
                if self.verbose:
                    print(f"Loaded entity data with {len(self.entity_data.get('movements', {}))} movements "
                          f"and {len(self.entity_data.get('politicians', {}))} politicians")
            except Exception as e:
                print(f"Error loading entity data: {e}")
        
        # Load word embeddings if provided
        if word_embedding_path and os.path.exists(word_embedding_path):
            try:
                self.word_embedding_store = ChunkedEmbeddingStore(
                    word_embedding_path, 
                    index_file=index_file,
                    cache_size=cache_size,
                    verbose=verbose
                )
            except Exception as e:
                print(f"Error loading word embeddings: {e}")
                self.word_embedding_store = None
    
    def get_entity_embedding(self, entity_type: str, entity_name: str) -> Optional[np.ndarray]:
        """
        Get the embedding for a specific entity.
        
        Args:
            entity_type: Type of entity ('movement' or 'politician')
            entity_name: Name of the entity
            
        Returns:
            Embedding vector or None if entity not found
        """
        if not self.entity_data:
            return None
        
        # Get entity data
        entity_data = None
        if entity_type == 'movement' and entity_name in self.entity_data.get('movements', {}):
            entity_data = self.entity_data['movements'][entity_name]
        elif entity_type == 'politician' and entity_name in self.entity_data.get('politicians', {}):
            entity_data = self.entity_data['politicians'][entity_name]
        else:
            return None
        
        # Get embedding from entity data
        if 'embedding' in entity_data:
            return np.array(entity_data['embedding'])
        
        return None
    
    def get_entity_position(self, entity_type: str, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the position data for a specific entity.
        
        Args:
            entity_type: Type of entity ('movement' or 'politician')
            entity_name: Name of the entity
            
        Returns:
            Position data or None if entity not found
        """
        if not self.entity_data:
            return None
        
        # Get entity data
        entity_data = None
        if entity_type == 'movement' and entity_name in self.entity_data.get('movements', {}):
            entity_data = self.entity_data['movements'][entity_name]
        elif entity_type == 'politician' and entity_name in self.entity_data.get('politicians', {}):
            entity_data = self.entity_data['politicians'][entity_name]
        else:
            return None
        
        # Get position data from entity data
        if 'position' in entity_data:
            return entity_data['position']
        
        return None
    
    def get_nearest_words_to_entity(self, 
                                   entity_type: str, 
                                   entity_name: str, 
                                   top_n: int = 50, 
                                   max_distance: float = 0.5) -> List[Dict[str, Any]]:
        """
        Find words closest to a specific entity in the embedding space.
        
        Args:
            entity_type: Type of entity ('movement' or 'politician')
            entity_name: Name of the entity
            top_n: Maximum number of words to return
            max_distance: Maximum distance threshold
            
        Returns:
            List of nearby words with distances and positions
        """
        if self.word_embedding_store is None:
            return []
        
        # Get entity embedding
        entity_embedding = self.get_entity_embedding(entity_type, entity_name)
        if entity_embedding is None:
            return []
        
        # Find nearest words
        return self.word_embedding_store.get_nearest_words(
            entity_embedding,
            top_n=top_n,
            max_distance=max_distance
        )
    
    def get_hybrid_visualization_data(self, 
                                     entity_type: Optional[str] = None,
                                     entity_name: Optional[str] = None,
                                     num_words: int = 100,
                                     max_distance: float = 0.5) -> Dict[str, Any]:
        """
        Generate data for a hybrid visualization combining entity-level and word-level data.
        
        Args:
            entity_type: Type of entity to focus on (optional)
            entity_name: Name of entity to focus on (optional)
            num_words: Number of words to include
            max_distance: Maximum distance for word inclusion
            
        Returns:
            Dictionary with visualization data
        """
        result = {
            'entities': self.get_entity_visualization_data(entity_type, entity_name),
            'words': []
        }
        
        # If we have a word embedding store and a focus entity, add related words
        if self.word_embedding_store is not None and entity_type and entity_name:
            entity_embedding = self.get_entity_embedding(entity_type, entity_name)
            if entity_embedding is not None:
                nearest_words = self.word_embedding_store.get_nearest_words(
                    entity_embedding,
                    top_n=num_words,
                    max_distance=max_distance
                )
                
                # Add words to result
                result['words'] = nearest_words
        
        return result
    
    def get_entity_visualization_data(self, 
                                     focus_entity_type: Optional[str] = None,
                                     focus_entity_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate entity-level visualization data.
        
        Args:
            focus_entity_type: Type of entity to focus on (optional)
            focus_entity_name: Name of entity to focus on (optional)
            
        Returns:
            Dictionary with entity visualization data
        """
        if not self.entity_data:
            return {'movements': [], 'politicians': []}
        
        result = {
            'movements': [],
            'politicians': []
        }
        
        # Process movements
        for name, data in self.entity_data.get('movements', {}).items():
            position = data.get('position', {})
            coords = position.get('coordinates', {})
            
            # Check if this is the focus entity
            is_focus = (focus_entity_type == 'movement' and focus_entity_name == name)
            
            movement_data = {
                'name': name,
                'x': coords.get('x', 0),
                'y': coords.get('y', 0),
                'size': position.get('size', 10),
                'color': position.get('color', 'rgba(255,0,0,0.7)'),
                'is_focus': is_focus
            }
            
            result['movements'].append(movement_data)
        
        # Process politicians
        for name, data in self.entity_data.get('politicians', {}).items():
            position = data.get('position', {})
            coords = position.get('coordinates', {})
            movement = data.get('movement', '')
            
            # Check if this is the focus entity
            is_focus = (focus_entity_type == 'politician' and focus_entity_name == name)
            
            politician_data = {
                'name': name,
                'x': coords.get('x', 0),
                'y': coords.get('y', 0),
                'size': position.get('size', 5),
                'color': position.get('color', 'rgba(0,0,255,0.7)'),
                'movement': movement,
                'is_focus': is_focus
            }
            
            result['politicians'].append(politician_data)
        
        return result
    
    def get_word_cloud_for_entity(self, 
                                 entity_type: str, 
                                 entity_name: str,
                                 top_n: int = 100,
                                 max_distance: float = 0.5) -> List[Dict[str, Any]]:
        """
        Generate word cloud data for a specific entity.
        
        Args:
            entity_type: Type of entity ('movement' or 'politician')
            entity_name: Name of the entity
            top_n: Number of words to include
            max_distance: Maximum distance threshold
            
        Returns:
            List of word cloud items
        """
        if self.word_embedding_store is None:
            return []
        
        # Get entity embedding
        entity_embedding = self.get_entity_embedding(entity_type, entity_name)
        if entity_embedding is None:
            return []
        
        # Find nearest words
        nearest_words = self.word_embedding_store.get_nearest_words(
            entity_embedding,
            top_n=top_n,
            max_distance=max_distance
        )
        
        # Format for word cloud
        word_cloud_data = [
            {
                'text': w['word'],
                'value': 1.0 - w['distance'],  # Convert distance to similarity
                'distance': w['distance']
            }
            for w in nearest_words
        ]
        
        return word_cloud_data
    
    def create_new_latent_space(self, 
                               entity_data: Dict[str, Any],
                               word_embeddings: Dict[str, np.ndarray],
                               output_dir: str,
                               entity_output_file: str = 'multi_level_analysis_results.json',
                               word_embedding_output_file: str = 'word_embeddings.h5',
                               index_type: str = 'flat') -> bool:
        """
        Create a new multi-level latent space from scratch.
        
        Args:
            entity_data: Dictionary with entity-level data
            word_embeddings: Dictionary mapping words to embedding vectors
            output_dir: Directory to save output files
            entity_output_file: Filename for entity data
            word_embedding_output_file: Filename for word embeddings
            index_type: Type of FAISS index to create ('flat', 'ivf', or 'hnsw')
            
        Returns:
            True if successful, False otherwise
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        entity_output_path = output_path / entity_output_file
        word_embedding_output_path = output_path / word_embedding_output_file
        index_output_path = word_embedding_output_path.with_suffix('.index')
        
        try:
            # Save entity data
            with open(entity_output_path, 'w', encoding='utf-8') as f:
                json.dump(entity_data, f, ensure_ascii=False, indent=2)
            
            if self.verbose:
                print(f"Saved entity data to {entity_output_path}")
            
            # Save word embeddings
            words = list(word_embeddings.keys())
            embeddings = np.array([word_embeddings[w] for w in words], dtype=np.float32)
            
            with h5py.File(word_embedding_output_path, 'w') as f:
                # Store words as dataset
                word_ds = f.create_dataset('words', data=words)
                
                # Store embeddings as dataset
                embedding_dim = embeddings.shape[1]
                embedding_ds = f.create_dataset(
                    'embeddings', 
                    shape=(len(words), embedding_dim),
                    dtype=np.float32
                )
                
                # Write in chunks to avoid memory issues
                chunk_size = 10000
                for i in range(0, len(words), chunk_size):
                    end_idx = min(i + chunk_size, len(words))
                    embedding_ds[i:end_idx] = embeddings[i:end_idx]
                
                # Add metadata
                f.attrs['num_words'] = len(words)
                f.attrs['embedding_dim'] = embedding_dim
            
            if self.verbose:
                print(f"Saved {len(words)} word embeddings to {word_embedding_output_path}")
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            
            if index_type == 'flat':
                index = faiss.IndexFlatL2(dimension)
            elif index_type == 'ivf':
                # IVF index needs to be trained on a sample
                nlist = min(4096, len(words) // 10)  # Number of clusters
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
                
                # Train on the data
                index.train(embeddings)
            elif index_type == 'hnsw':
                # HNSW index for faster search
                index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors per node
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
            
            # Add vectors to index
            chunk_size = 10000
            for i in range(0, len(words), chunk_size):
                end_idx = min(i + chunk_size, len(words))
                index.add(embeddings[i:end_idx])
            
            # Save the index
            faiss.write_index(index, str(index_output_path))
            
            if self.verbose:
                print(f"Created and saved FAISS index to {index_output_path}")
            
            return True
            
        except Exception as e:
            print(f"Error creating latent space: {e}")
            return False
    
    def close(self):
        """Close any open resources."""
        if self.word_embedding_store:
            self.word_embedding_store.close()
    
    def __del__(self):
        """Destructor to ensure resources are closed."""
        try:
            self.close()
        except:
            pass
