"""
Multi-level latent space analysis for political texts.

This module provides functionality to create and compare latent spaces
at different levels (e.g., party level and politician level).
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from .latent_space import PoliticalLatentSpace
from .embedding import EnhancedEmbedder


class MultiLevelLatentSpace:
    """
    Creates and manages multiple latent spaces at different levels
    (e.g., party level and politician level) for comparative analysis.
    """
    
    def __init__(self, embedder: EnhancedEmbedder):
        """
        Initialize the multi-level latent space.
        
        Args:
            embedder: An instance of EnhancedEmbedder
        """
        self.embedder = embedder
        self.party_latent_space = PoliticalLatentSpace(embedder)
        self.politician_latent_space = PoliticalLatentSpace(embedder)
        self.party_to_politicians = {}  # Mapping from parties to their politicians
        self.politician_to_party = {}   # Mapping from politicians to their party
    
    def define_anchors(self, anchors_dict: Dict[str, str]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Define semantic anchors for dimensions at both party and politician levels.
        
        Args:
            anchors_dict: Dictionary mapping dimension names to anchor texts
            
        Returns:
            Dictionary with party and politician anchor embeddings
        """
        party_anchors = self.party_latent_space.define_anchors(anchors_dict)
        politician_anchors = self.politician_latent_space.define_anchors(anchors_dict)
        
        return {
            "party": party_anchors,
            "politician": politician_anchors
        }
    
    def build_party_latent_space(self, party_texts: Dict[str, str], 
                                n_components: int = 8, 
                                method: str = 'umap') -> Dict[str, Any]:
        """
        Build the latent space for parties.
        
        Args:
            party_texts: Dictionary mapping party names to their text content
            n_components: Number of dimensions to learn
            method: Dimensionality reduction method ('umap', 'pca', or 'tsne')
            
        Returns:
            Result dictionary with success status and additional info
        """
        texts = list(party_texts.values())
        labels = list(party_texts.keys())
        
        result = self.party_latent_space.learn_dimensions(
            texts=texts,
            labels=labels,
            n_components=n_components,
            method=method,
            random_state=42
        )
        
        return result
    
    def build_politician_latent_space(self, politician_texts: Dict[str, Dict[str, Any]], 
                                     n_components: int = 8, 
                                     method: str = 'umap') -> Dict[str, Any]:
        """
        Build the latent space for politicians.
        
        Args:
            politician_texts: Dictionary mapping politician names to their data
                             (must include 'text' and 'party' keys)
            n_components: Number of dimensions to learn
            method: Dimensionality reduction method ('umap', 'pca', or 'tsne')
            
        Returns:
            Result dictionary with success status and additional info
        """
        texts = []
        labels = []
        
        # Update politician-party mappings
        for politician, data in politician_texts.items():
            if 'text' in data and 'party' in data:
                texts.append(data['text'])
                labels.append(politician)
                
                # Update mappings
                party = data['party']
                self.politician_to_party[politician] = party
                
                if party not in self.party_to_politicians:
                    self.party_to_politicians[party] = []
                
                self.party_to_politicians[party].append(politician)
        
        result = self.politician_latent_space.learn_dimensions(
            texts=texts,
            labels=labels,
            n_components=n_components,
            method=method,
            random_state=42
        )
        
        return result
    
    def position_party(self, party_name: str, party_text: str) -> Dict[str, Any]:
        """
        Position a party in the party latent space.
        
        Args:
            party_name: Name of the party
            party_text: Text content of the party
            
        Returns:
            Dictionary with expert axes and learned dimensions
        """
        return self.party_latent_space.position_text(party_text)
    
    def position_politician(self, politician_name: str, politician_text: str) -> Dict[str, Any]:
        """
        Position a politician in the politician latent space.
        
        Args:
            politician_name: Name of the politician
            politician_text: Text content of the politician
            
        Returns:
            Dictionary with expert axes and learned dimensions
        """
        return self.politician_latent_space.position_text(politician_text)
    
    def compare_politician_to_party(self, politician_name: str, party_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare a politician's position to their party's position.
        
        Args:
            politician_name: Name of the politician
            party_name: Name of the party (if None, will use the mapped party)
            
        Returns:
            Dictionary with comparison metrics
        """
        if politician_name not in self.politician_latent_space.corpus_labels:
            return {"error": f"Politician '{politician_name}' not found in latent space"}
        
        if party_name is None:
            if politician_name not in self.politician_to_party:
                return {"error": f"No party mapping found for politician '{politician_name}'"}
            party_name = self.politician_to_party[politician_name]
        
        if party_name not in self.party_latent_space.corpus_labels:
            return {"error": f"Party '{party_name}' not found in latent space"}
        
        # Get politician embedding
        politician_idx = self.politician_latent_space.corpus_labels.index(politician_name)
        politician_embedding = self.politician_latent_space.corpus_embeddings[politician_idx]
        
        # Get party embedding
        party_idx = self.party_latent_space.corpus_labels.index(party_name)
        party_embedding = self.party_latent_space.corpus_embeddings[party_idx]
        
        # Compare embeddings directly (raw embeddings, not reduced dimensions)
        similarity = cosine_similarity([politician_embedding], [party_embedding])[0][0]
        
        # Get positions on expert axes by manually calculating similarities with anchors
        politician_axes = {}
        party_axes = {}
        
        # Calculate similarities for politician
        politician_similarities = {}
        for dim, anchor_emb in self.politician_latent_space.anchor_embeddings.items():
            politician_similarities[dim] = float(cosine_similarity([politician_embedding], [anchor_emb])[0][0])
        
        # Calculate similarities for party
        party_similarities = {}
        for dim, anchor_emb in self.party_latent_space.anchor_embeddings.items():
            party_similarities[dim] = float(cosine_similarity([party_embedding], [anchor_emb])[0][0])
        
        # Calculate axes for politician
        if "economic_right" in politician_similarities and "economic_left" in politician_similarities:
            politician_axes["economic_axis"] = politician_similarities["economic_right"] - politician_similarities["economic_left"]
        
        if "progressive" in politician_similarities and "conservative" in politician_similarities:
            politician_axes["social_axis"] = politician_similarities["progressive"] - politician_similarities["conservative"]
        
        if "ecological" in politician_similarities and "growth" in politician_similarities:
            politician_axes["ecological_axis"] = politician_similarities["ecological"] - politician_similarities["growth"]
        
        if "nationalist" in politician_similarities and "internationalist" in politician_similarities:
            politician_axes["governance_axis"] = politician_similarities["nationalist"] - politician_similarities["internationalist"]
        
        # Calculate axes for party
        if "economic_right" in party_similarities and "economic_left" in party_similarities:
            party_axes["economic_axis"] = party_similarities["economic_right"] - party_similarities["economic_left"]
        
        if "progressive" in party_similarities and "conservative" in party_similarities:
            party_axes["social_axis"] = party_similarities["progressive"] - party_similarities["conservative"]
        
        if "ecological" in party_similarities and "growth" in party_similarities:
            party_axes["ecological_axis"] = party_similarities["ecological"] - party_similarities["growth"]
        
        if "nationalist" in party_similarities and "internationalist" in party_similarities:
            party_axes["governance_axis"] = party_similarities["nationalist"] - party_similarities["internationalist"]
        
        # Calculate differences on expert axes
        expert_axes_diff = {}
        for axis in politician_axes:
            if axis in party_axes:
                expert_axes_diff[axis] = politician_axes[axis] - party_axes[axis]
        
        return {
            "politician": politician_name,
            "party": party_name,
            "similarity": similarity,
            "expert_axes_diff": expert_axes_diff,
            # Note: We don't compare learned dimensions directly as they're in different spaces
        }
    
    def find_party_outliers(self, party_name: str, threshold: float = 0.2) -> Dict[str, Any]:
        """
        Find politicians who deviate significantly from their party's position.
        
        Args:
            party_name: Name of the party
            threshold: Similarity threshold below which a politician is considered an outlier
            
        Returns:
            Dictionary with outlier politicians and their deviation metrics
        """
        if party_name not in self.party_to_politicians:
            return {"error": f"No politicians found for party '{party_name}'"}
        
        party_politicians = self.party_to_politicians[party_name]
        outliers = []
        
        for politician in party_politicians:
            comparison = self.compare_politician_to_party(politician, party_name)
            
            if "error" not in comparison and comparison["similarity"] < (1 - threshold):
                outliers.append({
                    "politician": politician,
                    "similarity": comparison["similarity"],
                    "expert_axes_diff": comparison["expert_axes_diff"]
                })
        
        # Sort outliers by similarity (ascending)
        outliers.sort(key=lambda x: x["similarity"])
        
        return {
            "party": party_name,
            "outlier_count": len(outliers),
            "outliers": outliers
        }
    
    def calculate_party_cohesion(self, party_name: str) -> Dict[str, Any]:
        """
        Calculate the cohesion of a party based on the similarity of its politicians.
        
        Args:
            party_name: Name of the party
            
        Returns:
            Dictionary with cohesion metrics
        """
        if party_name not in self.party_to_politicians:
            return {"error": f"No politicians found for party '{party_name}'"}
        
        party_politicians = self.party_to_politicians[party_name]
        
        if len(party_politicians) < 2:
            return {
                "party": party_name,
                "cohesion": 1.0,  # Perfect cohesion with only one politician
                "politician_count": 1,
                "interpretation": "Perfect cohesion (only one politician)"
            }
        
        # Get politician embeddings
        politician_embeddings = []
        for politician in party_politicians:
            if politician in self.politician_latent_space.corpus_labels:
                idx = self.politician_latent_space.corpus_labels.index(politician)
                politician_embeddings.append(self.politician_latent_space.corpus_embeddings[idx])
        
        if len(politician_embeddings) < 2:
            return {"error": f"Not enough politician embeddings found for party '{party_name}'"}
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(politician_embeddings)
        
        # Calculate average similarity (excluding self-similarity)
        n = similarities.shape[0]
        total_similarity = 0
        count = 0
        
        for i in range(n):
            for j in range(i+1, n):  # Only upper triangle
                total_similarity += similarities[i, j]
                count += 1
        
        avg_similarity = total_similarity / count if count > 0 else 0
        
        return {
            "party": party_name,
            "cohesion": avg_similarity,
            "politician_count": len(politician_embeddings),
            "interpretation": self._interpret_cohesion(avg_similarity)
        }
    
    def _interpret_cohesion(self, cohesion: float) -> str:
        """
        Interpret the cohesion score.
        
        Args:
            cohesion: Cohesion score (0-1)
            
        Returns:
            String interpretation
        """
        if cohesion > 0.9:
            return "Very high cohesion - extremely unified party"
        elif cohesion > 0.8:
            return "High cohesion - strongly unified party"
        elif cohesion > 0.7:
            return "Moderate cohesion - generally unified party"
        elif cohesion > 0.6:
            return "Low cohesion - some internal differences"
        elif cohesion > 0.5:
            return "Very low cohesion - significant internal differences"
        else:
            return "Extremely low cohesion - highly fragmented party"
    
    def compare_all_parties(self) -> Dict[str, Any]:
        """
        Compare all parties to each other based on their positions.
        
        Returns:
            Dictionary with party comparison metrics
        """
        if self.party_latent_space.corpus_embeddings is None:
            return {"error": "Party latent space not built"}
        
        party_labels = self.party_latent_space.corpus_labels
        party_embeddings = self.party_latent_space.corpus_embeddings
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(party_embeddings)
        
        # Create comparison matrix
        comparison_matrix = {}
        for i, party1 in enumerate(party_labels):
            comparison_matrix[party1] = {}
            for j, party2 in enumerate(party_labels):
                comparison_matrix[party1][party2] = similarities[i, j]
        
        # Find most similar and most different pairs
        most_similar = {"pair": None, "similarity": -1}
        most_different = {"pair": None, "similarity": 2}
        
        for i, party1 in enumerate(party_labels):
            for j, party2 in enumerate(party_labels):
                if i < j:  # Only consider unique pairs
                    sim = similarities[i, j]
                    if sim > most_similar["similarity"]:
                        most_similar["similarity"] = sim
                        most_similar["pair"] = (party1, party2)
                    if sim < most_different["similarity"]:
                        most_different["similarity"] = sim
                        most_different["pair"] = (party1, party2)
        
        return {
            "comparison_matrix": comparison_matrix,
            "most_similar": most_similar,
            "most_different": most_different
        }
