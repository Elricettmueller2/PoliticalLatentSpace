"""
Latent space construction and manipulation for political text analysis.
"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, Any, List, Optional, Union
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class PoliticalLatentSpace:
    """
    Constructs and manages a latent space for political text analysis.
    Combines expert-defined dimensions with learned dimensions.
    """
    
    def __init__(self, embedder):
        """
        Initialize the political latent space.
        
        Args:
            embedder: An instance of EnhancedEmbedder
        """
        self.embedder = embedder
        self.anchors = {}
        self.anchor_embeddings = {}
        self.learned_dimensions = None
        self.dimension_reducer = None
        self.corpus_embeddings = None
        self.corpus_labels = None
    
    def define_anchors(self, anchors_dict: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Define semantic anchors for dimensions.
        
        Args:
            anchors_dict: Dictionary mapping dimension names to anchor texts
            
        Returns:
            Dictionary of anchor embeddings
        """
        self.anchors = anchors_dict
        print(f"Generating embeddings for {len(anchors_dict)} anchors...")
        
        # Generate embeddings for anchors
        self.anchor_embeddings = {
            k: self.embedder.encode(v) for k, v in anchors_dict.items()
        }
        
        print("Anchor embeddings generated.")
        return self.anchor_embeddings
    
    def position_text(self, text: str, use_learned: bool = True) -> Dict[str, Any]:
        """
        Position a text in the latent space.
        
        Args:
            text: Text to position
            use_learned: Whether to use learned dimensions (if available)
            
        Returns:
            Dictionary with position information
        """
        # Generate embedding for text
        embedding = self.embedder.encode(text)
        
        position = {}
        
        # Calculate position using anchor-based dimensions
        if self.anchor_embeddings:
            similarities = {
                dim: float(cosine_similarity([embedding], [anchor_emb])[0][0])
                for dim, anchor_emb in self.anchor_embeddings.items()
            }
            
            # Calculate positions on predefined axes
            axes = {}
            
            # Economic axis: right - left
            if "economic_right" in similarities and "economic_left" in similarities:
                axes["economic_axis"] = similarities["economic_right"] - similarities["economic_left"]
            
            # Social axis: progressive - conservative
            if "progressive" in similarities and "conservative" in similarities:
                axes["social_axis"] = similarities["progressive"] - similarities["conservative"]
            
            # Ecological axis: ecological - growth
            if "ecological" in similarities and "growth" in similarities:
                axes["ecological_axis"] = similarities["ecological"] - similarities["growth"]
            
            # Governance axis: nationalist - internationalist
            if "nationalist" in similarities and "internationalist" in similarities:
                axes["governance_axis"] = similarities["nationalist"] - similarities["internationalist"]
            
            position["expert_dimensions"] = {
                "axes": axes,
                "raw_similarities": similarities
            }
        
        # Use learned dimensions if available
        if use_learned and self.dimension_reducer is not None:
            try:
                learned_position = self.dimension_reducer.transform([embedding])[0]
                position["learned_dimensions"] = {
                    f"dim_{i}": float(val) for i, val in enumerate(learned_position)
                }
            except Exception as e:
                print(f"Error using learned dimensions: {e}")
                position["learned_dimensions"] = {}
        
        # Add raw embedding
        position["raw_embedding"] = embedding.tolist()
        
        return position
    
    def learn_dimensions(self, texts: List[str], labels: Optional[List[str]] = None, 
                        n_components: int = 8, method: str = 'umap',
                        random_state: int = 42) -> Dict[str, Any]:
        """
        Learn latent dimensions from a corpus of texts.
        
        Args:
            texts: List of texts to learn from
            labels: Optional list of labels for the texts
            n_components: Number of dimensions to learn
            method: Dimensionality reduction method ('umap', 'tsne', or 'pca')
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with learned dimensions information
        """
        if not texts:
            return {"success": False, "error": "No texts provided"}
        
        # Generate embeddings for all texts
        print(f"Generating embeddings for {len(texts)} texts...")
        self.corpus_embeddings = self.embedder.encode_batch(texts)
        self.corpus_labels = labels if labels else [f"Text_{i}" for i in range(len(texts))]
        
        # Apply dimensionality reduction
        print(f"Learning {n_components} dimensions using {method}...")
        
        try:
            if method.lower() == 'umap':
                # Parameters optimized for political text analysis:
                # - Higher n_neighbors preserves more global structure (political landscape)
                # - Lower min_dist creates tighter clusters of similar political entities
                # - Cosine metric is best for text embeddings
                self.dimension_reducer = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=30,       # Higher value (default is 15) for better global structure
                    min_dist=0.1,         # Lower value (default is 0.1) for tighter clusters
                    metric='cosine',
                    random_state=random_state,
                    low_memory=True       # More memory efficient for large datasets
                )
            elif method.lower() == 'tsne':
                self.dimension_reducer = TSNE(
                    n_components=n_components,
                    random_state=random_state,
                    metric='cosine'
                )
            elif method.lower() == 'pca':
                self.dimension_reducer = PCA(
                    n_components=n_components,
                    random_state=random_state
                )
            else:
                return {"success": False, "error": f"Unknown method: {method}"}
            
            self.learned_dimensions = self.dimension_reducer.fit_transform(self.corpus_embeddings)
            
            # Create a dictionary of learned dimensions
            learned_dict = {}
            for i, (label, position) in enumerate(zip(self.corpus_labels, self.learned_dimensions)):
                learned_dict[label] = {
                    f"dim_{j}": float(val) for j, val in enumerate(position)
                }
            
            print(f"Successfully learned {n_components} dimensions.")
            return {
                "success": True,
                "method": method,
                "n_components": n_components,
                "positions": learned_dict
            }
            
        except Exception as e:
            print(f"Error learning dimensions: {e}")
            return {"success": False, "error": str(e)}
    
    def compare_positions(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Compare the positions of two texts in the latent space.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary with comparison results
        """
        pos1 = self.position_text(text1)
        pos2 = self.position_text(text2)
        
        comparison = {"raw_similarity": 0.0}
        
        # Compare raw embeddings
        if "raw_embedding" in pos1 and "raw_embedding" in pos2:
            emb1 = np.array(pos1["raw_embedding"])
            emb2 = np.array(pos2["raw_embedding"])
            comparison["raw_similarity"] = float(cosine_similarity([emb1], [emb2])[0][0])
        
        # Compare expert dimensions
        if "expert_dimensions" in pos1 and "expert_dimensions" in pos2:
            exp1 = pos1["expert_dimensions"]
            exp2 = pos2["expert_dimensions"]
            
            # Compare axes
            if "axes" in exp1 and "axes" in exp2:
                axes_diff = {}
                for axis, val1 in exp1["axes"].items():
                    if axis in exp2["axes"]:
                        axes_diff[axis] = val1 - exp2["axes"][axis]
                
                comparison["axes_differences"] = axes_diff
            
            # Compare raw similarities
            if "raw_similarities" in exp1 and "raw_similarities" in exp2:
                sim_diff = {}
                for dim, val1 in exp1["raw_similarities"].items():
                    if dim in exp2["raw_similarities"]:
                        sim_diff[dim] = val1 - exp2["raw_similarities"][dim]
                
                comparison["similarity_differences"] = sim_diff
        
        # Compare learned dimensions
        if "learned_dimensions" in pos1 and "learned_dimensions" in pos2:
            ld1 = pos1["learned_dimensions"]
            ld2 = pos2["learned_dimensions"]
            
            ld_diff = {}
            for dim, val1 in ld1.items():
                if dim in ld2:
                    ld_diff[dim] = val1 - ld2[dim]
            
            comparison["learned_dimension_differences"] = ld_diff
        
        return comparison
    
    def position_on_axis(self, text: str, positive_anchor: str, negative_anchor: str) -> float:
        """
        Position a text on a custom axis defined by two anchor points.
        
        Args:
            text: Text to position
            positive_anchor: Text defining the positive end of the axis
            negative_anchor: Text defining the negative end of the axis
            
        Returns:
            Position on axis from -1.0 to 1.0 (0.0 is neutral)
        """
        # Generate embeddings
        text_embedding = self.embedder.encode(text)
        pos_embedding = self.embedder.encode(positive_anchor)
        neg_embedding = self.embedder.encode(negative_anchor)
        
        # Calculate similarities
        pos_sim = float(cosine_similarity([text_embedding], [pos_embedding])[0][0])
        neg_sim = float(cosine_similarity([text_embedding], [neg_embedding])[0][0])
        
        # Calculate position (normalized to -1 to 1 range)
        # Higher value means closer to positive anchor
        position = pos_sim - neg_sim
        
        return position
    
    def get_nearest_texts(self, text: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Find the nearest texts in the corpus to the given text.
        
        Args:
            text: Query text
            top_n: Number of nearest texts to return
            
        Returns:
            List of nearest texts with similarity scores
        """
        if self.corpus_embeddings is None or self.corpus_labels is None:
            return []
        
        # Generate embedding for query text
        query_embedding = self.embedder.encode(text)
        
        # Calculate similarities to all corpus texts
        similarities = cosine_similarity([query_embedding], self.corpus_embeddings)[0]
        
        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1]  # Descending order
        
        # Get top N results
        results = []
        for i in sorted_indices[:top_n]:
            results.append({
                "label": self.corpus_labels[i],
                "similarity": float(similarities[i])
            })
        
        return results
    
    def evaluate_learned_dimensions(self, test_texts: Dict[str, str], 
                                  known_groups: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Evaluate how well the learned dimensions separate known political groups.
        
        Args:
            test_texts: Dictionary mapping entity names to their texts
            known_groups: Dictionary mapping group names to lists of entity names
                e.g., {'left_wing': ['spd', 'linke'], 'right_wing': ['cdu', 'afd']}
            
        Returns:
            Evaluation metrics including separation ratio and quality assessment
        """
        results = {}
        
        # Position all test texts
        positions = {}
        for name, text in test_texts.items():
            pos = self.position_text(text)
            if 'learned_dimensions' in pos:
                positions[name] = pos['learned_dimensions']
        
        if not positions:
            return {"error": "No learned dimensions available or no positions could be calculated"}
        
        # Calculate in-group vs. out-group distances
        in_group_distances = []
        out_group_distances = []
        
        for group_name, members in known_groups.items():
            # Get positions for group members
            group_positions = [positions[m] for m in members if m in positions]
            if len(group_positions) < 2:
                continue
                
            # Calculate in-group distances
            for i in range(len(group_positions)):
                for j in range(i+1, len(group_positions)):
                    pos1 = np.array(list(group_positions[i].values()))
                    pos2 = np.array(list(group_positions[j].values()))
                    dist = np.linalg.norm(pos1 - pos2)
                    in_group_distances.append(dist)
            
            # Calculate out-group distances
            other_members = [m for m in positions.keys() if m not in members]
            other_positions = [positions[m] for m in other_members if m in positions]
            
            for group_pos in group_positions:
                for other_pos in other_positions:
                    pos1 = np.array(list(group_pos.values()))
                    pos2 = np.array(list(other_pos.values()))
                    dist = np.linalg.norm(pos1 - pos2)
                    out_group_distances.append(dist)
        
        # Calculate metrics
        if in_group_distances and out_group_distances:
            avg_in_group = sum(in_group_distances) / len(in_group_distances)
            avg_out_group = sum(out_group_distances) / len(out_group_distances)
            separation_ratio = avg_out_group / avg_in_group
            
            results["avg_in_group_distance"] = avg_in_group
            results["avg_out_group_distance"] = avg_out_group
            results["separation_ratio"] = separation_ratio
            results["quality"] = "good" if separation_ratio > 1.5 else "fair" if separation_ratio > 1.2 else "poor"
            
            # Calculate per-group metrics
            group_metrics = {}
            for group_name, members in known_groups.items():
                valid_members = [m for m in members if m in positions]
                if len(valid_members) < 2:
                    continue
                    
                group_positions = [positions[m] for m in valid_members]
                
                # Calculate cohesion (average distance between members)
                distances = []
                for i in range(len(group_positions)):
                    for j in range(i+1, len(group_positions)):
                        pos1 = np.array(list(group_positions[i].values()))
                        pos2 = np.array(list(group_positions[j].values()))
                        dist = np.linalg.norm(pos1 - pos2)
                        distances.append(dist)
                
                avg_distance = sum(distances) / len(distances) if distances else 0
                group_metrics[group_name] = {
                    "cohesion": avg_distance,
                    "member_count": len(valid_members)
                }
            
            results["group_metrics"] = group_metrics
        else:
            results["error"] = "Insufficient data to calculate metrics"
        
        return results
