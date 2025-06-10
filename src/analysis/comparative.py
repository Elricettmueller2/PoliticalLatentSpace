"""
Comparative analysis functionality for political texts.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ComparativeAnalyzer:
    """
    Performs comparative analysis between political movements.
    """
    
    def __init__(self, latent_space, term_analyzer):
        """
        Initialize the comparative analyzer.
        
        Args:
            latent_space: An instance of PoliticalLatentSpace
            term_analyzer: An instance of TermUsageAnalyzer
        """
        self.latent_space = latent_space
        self.term_analyzer = term_analyzer
    
    def compare_movements(self, texts_dict: Dict[str, str]) -> Dict[str, Any]:
        """
        Compare multiple political movements.
        
        Args:
            texts_dict: Dictionary mapping movement names to text content
            
        Returns:
            Dictionary with comparison results
        """
        if not texts_dict or len(texts_dict) < 2:
            return {"error": "Need at least two movements to compare"}
        
        # Get all movement names
        movements = list(texts_dict.keys())
        
        # Compare each pair of movements
        pairwise_comparisons = {}
        for i, movement1 in enumerate(movements):
            for movement2 in movements[i+1:]:
                comparison = self.compare_two_movements(
                    movement1, texts_dict[movement1],
                    movement2, texts_dict[movement2]
                )
                pairwise_comparisons[f"{movement1}_vs_{movement2}"] = comparison
        
        # Find common and distinctive terms across all movements
        common_terms = self.find_common_terms(texts_dict)
        distinctive_terms = self.find_distinctive_terms_all(texts_dict)
        
        return {
            "movements": movements,
            "pairwise_comparisons": pairwise_comparisons,
            "common_terms": common_terms,
            "distinctive_terms": distinctive_terms
        }
    
    def compare_two_movements(self, name1: str, text1: str, 
                             name2: str, text2: str) -> Dict[str, Any]:
        """
        Compare two political movements.
        
        Args:
            name1: Name of the first movement
            text1: Text of the first movement
            name2: Name of the second movement
            text2: Text of the second movement
            
        Returns:
            Dictionary with comparison results
        """
        # Position both texts in latent space
        pos1 = self.latent_space.position_text(text1)
        pos2 = self.latent_space.position_text(text2)
        
        # Compare positions
        position_comparison = self.latent_space.compare_positions(text1, text2)
        
        # Find distinctive terms
        distinctive_terms = self.term_analyzer.find_distinctive_terms(text1, text2)
        
        # Compare term usage for common terms
        term_usage_comparison = {}
        
        # Get top terms from both texts
        terms1 = [term for term, _ in self.term_analyzer.extract_key_terms(text1)]
        terms2 = [term for term, _ in self.term_analyzer.extract_key_terms(text2)]
        
        # Find common terms
        common_terms = list(set(terms1[:10]) & set(terms2[:10]))
        
        # Compare usage of common terms
        for term in common_terms[:5]:  # Limit to 5 terms for efficiency
            comparison = self.term_analyzer.compare_term_usage_across_texts(
                {name1: text1, name2: text2}, term
            )
            term_usage_comparison[term] = comparison
        
        return {
            "position_comparison": position_comparison,
            "distinctive_terms": distinctive_terms,
            "common_terms": common_terms,
            "term_usage_comparison": term_usage_comparison
        }
    
    def find_common_terms(self, texts_dict: Dict[str, str], 
                         top_n: int = 10) -> List[str]:
        """
        Find terms common across multiple movements.
        
        Args:
            texts_dict: Dictionary mapping movement names to text content
            top_n: Number of top terms to consider from each text
            
        Returns:
            List of common terms
        """
        if not texts_dict or len(texts_dict) < 2:
            return []
        
        # Extract top terms for each text
        all_top_terms = {}
        for name, text in texts_dict.items():
            terms = self.term_analyzer.extract_key_terms(text, top_n=top_n*2)
            all_top_terms[name] = [term for term, _ in terms[:top_n]]
        
        # Find terms that appear in all texts
        movements = list(texts_dict.keys())
        common_terms = set(all_top_terms[movements[0]])
        
        for movement in movements[1:]:
            common_terms &= set(all_top_terms[movement])
        
        return list(common_terms)
    
    def find_distinctive_terms_all(self, texts_dict: Dict[str, str], 
                                  top_n: int = 5) -> Dict[str, List[str]]:
        """
        Find distinctive terms for each movement compared to all others.
        
        Args:
            texts_dict: Dictionary mapping movement names to text content
            top_n: Number of top distinctive terms to find
            
        Returns:
            Dictionary mapping movement names to their distinctive terms
        """
        if not texts_dict or len(texts_dict) < 2:
            return {}
        
        distinctive_terms = {}
        
        for name, text in texts_dict.items():
            # Combine all other texts
            other_texts = " ".join([t for n, t in texts_dict.items() if n != name])
            
            # Find distinctive terms
            comparison = self.term_analyzer.find_distinctive_terms(text, other_texts, top_n=top_n)
            distinctive_terms[name] = [term for term, _ in comparison["text1"]]
        
        return distinctive_terms
    
    def calculate_movement_similarity_matrix(self, texts_dict: Dict[str, str]) -> Dict[str, Any]:
        """
        Calculate a similarity matrix between movements.
        
        Args:
            texts_dict: Dictionary mapping movement names to text content
            
        Returns:
            Dictionary with similarity matrix and movement names
        """
        if not texts_dict or len(texts_dict) < 2:
            return {"error": "Need at least two movements"}
        
        movements = list(texts_dict.keys())
        embeddings = []
        
        # Generate embeddings for each text
        for name in movements:
            embedding = self.latent_space.embedder.encode(texts_dict[name])
            embeddings.append(embedding)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        return {
            "movements": movements,
            "similarity_matrix": similarity_matrix.tolist()
        }
    
    def find_closest_movements(self, texts_dict: Dict[str, str]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find the closest movement for each movement.
        
        Args:
            texts_dict: Dictionary mapping movement names to text content
            
        Returns:
            Dictionary mapping each movement to its closest movements
        """
        if not texts_dict or len(texts_dict) < 2:
            return {}
        
        similarity_data = self.calculate_movement_similarity_matrix(texts_dict)
        
        if "error" in similarity_data:
            return {}
        
        movements = similarity_data["movements"]
        similarity_matrix = np.array(similarity_data["similarity_matrix"])
        
        closest_movements = {}
        
        for i, movement in enumerate(movements):
            # Get similarities to all other movements
            similarities = [(movements[j], float(similarity_matrix[i, j])) 
                           for j in range(len(movements)) if i != j]
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            closest_movements[movement] = similarities
        
        return closest_movements
