"""
Context window analysis for political texts.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class ContextWindowAnalyzer:
    """
    Analyzes how specific terms are used in context within political texts.
    """
    
    def __init__(self, embedder, window_sizes=[30, 50, 100]):
        """
        Initialize the context analyzer.
        
        Args:
            embedder: An instance of EnhancedEmbedder
            window_sizes: List of window sizes (in characters) to use for context extraction
        """
        self.embedder = embedder
        self.window_sizes = window_sizes
    
    def extract_contexts(self, text: str, term: str, window_size: int) -> List[Dict[str, Any]]:
        """
        Extract context windows around term occurrences.
        
        Args:
            text: The text to analyze
            term: The term to find in the text
            window_size: Size of the context window (characters before and after the term)
            
        Returns:
            List of dictionaries containing context information
        """
        if not text or not term:
            return []
        
        # Find all occurrences of the term (case-insensitive)
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        matches = list(pattern.finditer(text))
        
        contexts = []
        for match in matches:
            start = max(0, match.start() - window_size)
            end = min(len(text), match.end() + window_size)
            
            # Extract the context
            context = text[start:end]
            
            # Find the position of the term within the context
            if start == 0:
                term_pos_in_context = match.start()
            else:
                term_pos_in_context = match.start() - start
            
            # Get some text before and after for easier analysis
            before_text = text[start:match.start()]
            after_text = text[match.end():end]
            
            contexts.append({
                "text": context,
                "position": match.start(),
                "term_position_in_context": term_pos_in_context,
                "before_text": before_text,
                "after_text": after_text,
                "term": text[match.start():match.end()]  # The actual term as it appears in text
            })
        
        return contexts
    
    def analyze_term_in_context(self, text: str, term: str, 
                               specific_window_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze how a term is used in different contexts within text.
        
        Args:
            text: The text to analyze
            term: The term to analyze
            specific_window_size: Optional specific window size to use
            
        Returns:
            Dictionary with analysis results
        """
        window_sizes = [specific_window_size] if specific_window_size else self.window_sizes
        results = {}
        
        for size in window_sizes:
            contexts = self.extract_contexts(text, term, size)
            if contexts:
                # Generate embeddings for each context
                context_texts = [c["text"] for c in contexts]
                context_embeddings = self.embedder.encode_batch(context_texts)
                
                # Add embeddings to contexts
                for i, context in enumerate(contexts):
                    context["embedding"] = context_embeddings[i].tolist()
                
                results[size] = contexts
        
        return {
            "term": term,
            "occurrences": sum(len(contexts) for contexts in results.values()),
            "contexts_by_window_size": results
        }
    
    def compare_contexts(self, contexts1: List[Dict[str, Any]], 
                        contexts2: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare how a term is used in two different sets of contexts.
        
        Args:
            contexts1: First set of contexts
            contexts2: Second set of contexts
            
        Returns:
            Dictionary with comparison results
        """
        if not contexts1 or not contexts2:
            return {"similarity": 0.0, "comparable_contexts": 0}
        
        # Extract embeddings
        embeddings1 = np.array([c["embedding"] for c in contexts1 if "embedding" in c])
        embeddings2 = np.array([c["embedding"] for c in contexts2 if "embedding" in c])
        
        if len(embeddings1) == 0 or len(embeddings2) == 0:
            return {"similarity": 0.0, "comparable_contexts": 0}
        
        # Calculate pairwise similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings1, embeddings2)
        
        # Calculate average similarity
        avg_similarity = float(np.mean(similarities))
        max_similarity = float(np.max(similarities))
        
        # Find most similar contexts
        most_similar_pair = np.unravel_index(np.argmax(similarities), similarities.shape)
        
        return {
            "average_similarity": avg_similarity,
            "max_similarity": max_similarity,
            "comparable_contexts": len(embeddings1) * len(embeddings2),
            "most_similar_pair": {
                "context1_index": int(most_similar_pair[0]),
                "context2_index": int(most_similar_pair[1]),
                "similarity": float(similarities[most_similar_pair])
            }
        }
    
    def find_semantic_clusters(self, contexts: List[Dict[str, Any]], 
                              n_clusters: int = 3) -> Dict[str, Any]:
        """
        Find semantic clusters in contexts to identify different usages of a term.
        
        Args:
            contexts: List of contexts
            n_clusters: Number of clusters to find
            
        Returns:
            Dictionary with clustering results
        """
        if not contexts or len(contexts) < n_clusters:
            return {"clusters": [], "success": False}
        
        # Extract embeddings
        embeddings = np.array([c["embedding"] for c in contexts if "embedding" in c])
        
        if len(embeddings) < n_clusters:
            return {"clusters": [], "success": False}
        
        # Perform clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Group contexts by cluster
        clustered_contexts = {}
        for i, cluster_id in enumerate(clusters):
            cluster_id = int(cluster_id)
            if cluster_id not in clustered_contexts:
                clustered_contexts[cluster_id] = []
            
            # Add index to context for reference
            context_copy = contexts[i].copy()
            context_copy["cluster_index"] = i
            clustered_contexts[cluster_id].append(context_copy)
        
        return {
            "clusters": clustered_contexts,
            "n_clusters": len(clustered_contexts),
            "success": True
        }
