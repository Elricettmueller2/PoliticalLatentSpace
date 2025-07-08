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
                                 max_distance: float = 5.0) -> List[Dict[str, Any]]:
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
        
        # Normalize the entity embedding to have the same magnitude as word embeddings
        entity_norm = np.linalg.norm(entity_embedding)
        if entity_norm > 0:
            # Scale up to have norm around 1.7 (similar to word embeddings)
            entity_embedding = entity_embedding * (1.7 / entity_norm)
            if self.verbose:
                print(f"Normalized entity embedding norm: {np.linalg.norm(entity_embedding)}")
        
        # Find nearest words - request more than we need so we can filter
        nearest_words = self.word_embedding_store.get_nearest_words(
            entity_embedding,
            top_n=top_n * 2,  # Get more words than needed for filtering
            max_distance=max_distance
        )
        
        # Filter out words that are likely not relevant to politics
        # Common German political terms and concepts
        political_prefixes = [
            'politik', 'demokrat', 'sozial', 'wirtschaft', 'recht', 'gesetz', 'staat', 
            'partei', 'wahl', 'bürger', 'regierung', 'opposition', 'parlament', 'bundestag',
            'reform', 'steuer', 'arbeit', 'umwelt', 'klima', 'migration', 'flüchtling',
            'europa', 'nation', 'international', 'sicherheit', 'freiheit', 'gerechtigkeit'
        ]
        
        # Prioritize words that match political prefixes but don't exclude others completely
        prioritized_words = []
        other_words = []
        
        for word_data in nearest_words:
            word = word_data['word'].lower()
            is_political = False
            
            # Check if the word starts with any political prefix
            for prefix in political_prefixes:
                if word.startswith(prefix) or prefix in word:
                    is_political = True
                    break
            
            if is_political:
                prioritized_words.append(word_data)
            else:
                other_words.append(word_data)
        
        # Combine prioritized words with other words, up to top_n total
        # Ensure we always include at least 50% of top_n words regardless of political relevance
        min_other_words = max(top_n // 2, top_n - len(prioritized_words))
        filtered_words = prioritized_words + other_words[:min_other_words]
        filtered_words = filtered_words[:top_n]
        
        if self.verbose:
            print(f"Found {len(prioritized_words)} political terms out of {len(nearest_words)} total words")
        
        # Format for word cloud with more meaningful value scaling
        word_cloud_data = []
        max_similarity = 0
        min_similarity = 1
        
        # First pass to find min and max similarity values
        for w in filtered_words:
            similarity = 1.0 - (w['distance'] / max_distance)
            max_similarity = max(max_similarity, similarity)
            min_similarity = min(min_similarity, similarity)
        
        # Normalize values to ensure better distribution
        similarity_range = max_similarity - min_similarity
        
        # Second pass to create word cloud data with normalized values
        for w in filtered_words:
            similarity = 1.0 - (w['distance'] / max_distance)
            
            # Normalize to 0-1 range and then scale to 0.3-1.0 range to ensure visibility
            normalized_value = 0.3
            if similarity_range > 0:
                normalized_value = 0.3 + 0.7 * ((similarity - min_similarity) / similarity_range)
            
            # Generate meaningful position values based on the word's embedding
            # Default to a random position if we can't calculate a meaningful one
            # Use a deterministic seed based on the word to ensure consistency
            word_lower = w['word'].lower()
            word_hash = sum(ord(c) * (i+1) for i, c in enumerate(word_lower))
            np.random.seed(word_hash)
            
            # Default random position with good spread
            position = [
                0.2 + 0.6 * np.random.random(),  # Economic axis (0.2-0.8)
                0.2 + 0.6 * np.random.random(),  # Social axis (0.2-0.8)
                0.2 + 0.6 * np.random.random()   # Ecological axis (0.2-0.8)
            ]

            if self.word_embedding_store is not None:
                word_embedding = self.word_embedding_store.get_word_embedding(word_lower)
                
                if word_embedding is not None and entity_embedding is not None:
                    # Calculate the projection of the word embedding onto the entity embedding
                    entity_norm = np.linalg.norm(entity_embedding)
                    word_norm = np.linalg.norm(word_embedding)
                    
                    if entity_norm > 0 and word_norm > 0:
                        # Calculate cosine similarity for the base relationship
                        cos_sim = np.dot(word_embedding, entity_embedding) / (word_norm * entity_norm)
                        
                        # Create a PCA-like approach for finding principal directions
                        # First axis is the entity direction
                        axis1 = entity_embedding / entity_norm
                        
                        # Find a component of the word embedding orthogonal to axis1
                        word_proj = np.dot(word_embedding, axis1) * axis1
                        ortho_component = word_embedding - word_proj
                        ortho_norm = np.linalg.norm(ortho_component)
                        
                        if ortho_norm > 1e-6:
                            # Second axis is the orthogonal component direction
                            axis2 = ortho_component / ortho_norm
                            
                            # Third axis is orthogonal to both
                            axis3 = np.cross(axis1[:3], axis2[:3])  # Use first 3 dimensions for cross product
                            if len(axis3) < len(axis1):
                                # Pad axis3 if needed
                                axis3 = np.pad(axis3, (0, len(axis1) - len(axis3)))
                            
                            axis3_norm = np.linalg.norm(axis3)
                            if axis3_norm > 1e-6:
                                axis3 = axis3 / axis3_norm
                            else:
                                # Fallback to a random orthogonal vector
                                axis3 = np.random.randn(len(axis1))
                                axis3 = axis3 - np.dot(axis3, axis1) * axis1
                                axis3 = axis3 - np.dot(axis3, axis2) * axis2
                                axis3_norm = np.linalg.norm(axis3)
                                if axis3_norm > 1e-6:
                                    axis3 = axis3 / axis3_norm
                        else:
                            # If orthogonal component is too small, create random orthogonal axes
                            # Create a random vector
                            axis2 = np.random.randn(len(axis1))
                            # Make it orthogonal to axis1
                            axis2 = axis2 - np.dot(axis2, axis1) * axis1
                            axis2_norm = np.linalg.norm(axis2)
                            if axis2_norm > 1e-6:
                                axis2 = axis2 / axis2_norm
                            
                            # Create third orthogonal axis
                            axis3 = np.cross(axis1[:3], axis2[:3])
                            if len(axis3) < len(axis1):
                                axis3 = np.pad(axis3, (0, len(axis1) - len(axis3)))
                            axis3_norm = np.linalg.norm(axis3)
                            if axis3_norm > 1e-6:
                                axis3 = axis3 / axis3_norm
                        
                        # Project the word embedding onto these axes
                        proj1 = np.dot(word_embedding, axis1)
                        proj2 = np.dot(word_embedding, axis2)
                        proj3 = np.dot(word_embedding, axis3)
                        
                        # Apply a non-linear transformation to spread out values
                        # and normalize to [0,1] range for visualization
                        position = [
                            (np.tanh(proj1 * 2) + 1) / 2,  # Economic axis (0-1)
                            (np.tanh(proj2 * 2) + 1) / 2,  # Social axis (0-1)
                            (np.tanh(proj3 * 2) + 1) / 2   # Ecological axis (0-1)
                        ]
            
            word_cloud_data.append({
                'text': w['word'],
                'value': normalized_value,
                'distance': w['distance'],
                'raw_similarity': similarity,
                'position': position
            })
        
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
