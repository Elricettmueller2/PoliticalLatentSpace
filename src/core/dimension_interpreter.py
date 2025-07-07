"""
Utilities for interpreting and analyzing learned dimensions in political latent space.
"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, Any, List, Optional, Union
from scipy.stats import pearsonr


class DimensionInterpreter:
    """
    Tools for interpreting and analyzing learned dimensions in political latent space.
    """
    
    def __init__(self, latent_space):
        """
        Initialize the dimension interpreter.
        
        Args:
            latent_space: A PoliticalLatentSpace instance
        """
        self.latent_space = latent_space
    
    def interpret_dimensions(self, reference_texts: Dict[str, str]) -> Dict[str, Any]:
        """
        Interpret what each learned dimension represents by correlating with reference texts.
        
        Args:
            reference_texts: Dict mapping concept names to representative texts
                e.g., {'left_economic': 'text about socialism...', 'right_economic': 'text about free markets...'}
        
        Returns:
            Dictionary mapping dimension indices to their most correlated concepts
        """
        if (self.latent_space.learned_dimensions is None or 
            self.latent_space.corpus_embeddings is None or
            len(self.latent_space.learned_dimensions) == 0):
            return {"error": "No learned dimensions available"}
        
        # Encode reference texts
        reference_embeddings = {}
        for concept, text in reference_texts.items():
            reference_embeddings[concept] = self.latent_space.embedder.encode(text)
        
        # For each learned dimension, find which reference concepts it correlates with
        dimension_interpretations = {}
        
        for dim_idx in range(self.latent_space.learned_dimensions.shape[1]):
            # Extract values for this dimension
            dim_values = self.latent_space.learned_dimensions[:, dim_idx]
            
            correlations = {}
            for concept, ref_embedding in reference_embeddings.items():
                # Project all corpus embeddings onto the reference embedding
                similarities = [cosine_similarity([self.latent_space.corpus_embeddings[i]], [ref_embedding])[0][0] 
                               for i in range(len(self.latent_space.corpus_embeddings))]
                
                # Calculate correlation between dimension values and similarities
                corr, p_value = pearsonr(dim_values, similarities)
                correlations[concept] = (corr, p_value)
            
            # Sort by absolute correlation
            sorted_correlations = sorted(correlations.items(), 
                                        key=lambda x: abs(x[1][0]), 
                                        reverse=True)
            
            # Store top correlations
            dimension_interpretations[f"dim_{dim_idx}"] = {
                'top_concepts': [(concept, float(corr), float(p_val)) for concept, (corr, p_val) in sorted_correlations[:3]],
                'interpretation': sorted_correlations[0][0] if sorted_correlations and abs(sorted_correlations[0][1][0]) > 0.3 else 'Unknown'
            }
        
        return dimension_interpretations
    
    def dimension_importance(self) -> Dict[str, float]:
        """
        Calculate the importance of each dimension based on variance explained.
        
        Returns:
            Dictionary mapping dimension indices to their importance scores
        """
        if not self.latent_space.learned_dimensions:
            return {"error": "No learned dimensions available"}
        
        # Calculate variance along each dimension
        variances = np.var(self.latent_space.learned_dimensions, axis=0)
        total_variance = np.sum(variances)
        
        # Calculate importance as percentage of variance explained
        importance = {}
        for i, var in enumerate(variances):
            importance[f"dim_{i}"] = float(var / total_variance)
        
        return importance
    
    def find_exemplar_texts(self, top_n: int = 3) -> Dict[str, List[str]]:
        """
        Find exemplar texts for each dimension (texts with highest/lowest values).
        
        Args:
            top_n: Number of exemplars to find for each extreme
            
        Returns:
            Dictionary mapping dimensions to lists of exemplar texts
        """
        if (self.latent_space.learned_dimensions is None or 
            self.latent_space.corpus_labels is None or 
            len(self.latent_space.learned_dimensions) == 0):
            return {"error": "No learned dimensions available"}
        
        exemplars = {}
        
        for dim_idx in range(self.latent_space.learned_dimensions.shape[1]):
            # Extract values for this dimension
            dim_values = self.latent_space.learned_dimensions[:, dim_idx]
            
            # Sort by dimension value
            sorted_indices = np.argsort(dim_values)
            
            # Get top and bottom exemplars
            low_indices = sorted_indices[:top_n]
            high_indices = sorted_indices[-top_n:][::-1]
            
            low_exemplars = [(self.latent_space.corpus_labels[i], float(dim_values[i])) for i in low_indices]
            high_exemplars = [(self.latent_space.corpus_labels[i], float(dim_values[i])) for i in high_indices]
            
            exemplars[f"dim_{dim_idx}"] = {
                "high_exemplars": high_exemplars,
                "low_exemplars": low_exemplars
            }
        
        return exemplars
    
    def dimension_correlation_matrix(self) -> Dict[str, Any]:
        """
        Calculate correlation matrix between dimensions to identify redundancy.
        
        Returns:
            Dictionary with correlation matrix and highly correlated pairs
        """
        if self.latent_space.learned_dimensions is None or len(self.latent_space.learned_dimensions) == 0:
            return {"error": "No learned dimensions available"}
        
        n_dims = self.latent_space.learned_dimensions.shape[1]
        corr_matrix = np.zeros((n_dims, n_dims))
        
        # Calculate correlation between each pair of dimensions
        for i in range(n_dims):
            for j in range(n_dims):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    corr, _ = pearsonr(
                        self.latent_space.learned_dimensions[:, i],
                        self.latent_space.learned_dimensions[:, j]
                    )
                    corr_matrix[i, j] = corr
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(n_dims):
            for j in range(i+1, n_dims):
                if abs(corr_matrix[i, j]) > 0.7:  # Threshold for high correlation
                    high_corr_pairs.append({
                        "dim1": f"dim_{i}",
                        "dim2": f"dim_{j}",
                        "correlation": float(corr_matrix[i, j])
                    })
        
        # Convert correlation matrix to dictionary
        corr_dict = {}
        for i in range(n_dims):
            corr_dict[f"dim_{i}"] = {f"dim_{j}": float(corr_matrix[i, j]) for j in range(n_dims)}
        
        return {
            "correlation_matrix": corr_dict,
            "highly_correlated_pairs": high_corr_pairs
        }
