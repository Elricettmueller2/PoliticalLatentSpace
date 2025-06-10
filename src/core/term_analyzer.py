"""
Term usage analysis for political texts.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import re


class TermUsageAnalyzer:
    """
    Analyzes how terms are used in political texts, including key term extraction
    and comparative analysis of term usage across different texts.
    """
    
    def __init__(self, context_analyzer, latent_space):
        """
        Initialize the term usage analyzer.
        
        Args:
            context_analyzer: An instance of ContextWindowAnalyzer
            latent_space: An instance of PoliticalLatentSpace
        """
        self.context_analyzer = context_analyzer
        self.latent_space = latent_space
        self._stopwords = self._get_stopwords()
    
    def extract_key_terms(self, text: str, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Extract key terms from text using TF-IDF.
        
        Args:
            text: Text to analyze
            top_n: Number of top terms to extract
            
        Returns:
            List of (term, score) tuples
        """
        if not text:
            return []
        
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=top_n*5,  # Extract more than we need, then filter
            stop_words=self._stopwords,
            ngram_range=(1, 3),    # Include single words, bigrams, and trigrams
            # No min_df parameter for small corpus
            max_df=1.0             # Allow terms that appear in all documents
        )
        
        try:
            # Generate TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform([text])
            
            # Get feature names and scores
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Sort by score
            sorted_items = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
            
            # Filter out terms that are too short or contain non-alphabetic characters
            filtered_items = [
                (term, score) for term, score in sorted_items
                if len(term) > 2 and not re.search(r'^\d+$', term)
            ]
            
            return filtered_items[:top_n]
        
        except Exception as e:
            print(f"Error extracting key terms: {e}")
            return []
    
    def analyze_term_usage(self, text: str, term: str) -> Dict[str, Any]:
        """
        Analyze how a term is used in a text.
        
        Args:
            text: Text to analyze
            term: Term to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Get context analysis
        context_analysis = self.context_analyzer.analyze_term_in_context(text, term)
        
        # If no occurrences, return empty analysis
        if context_analysis["occurrences"] == 0:
            return {
                "term": term,
                "occurrences": 0,
                "contexts": []
            }
        
        # Get the largest window size with contexts
        window_sizes = sorted(context_analysis["contexts_by_window_size"].keys())
        if not window_sizes:
            return {
                "term": term,
                "occurrences": 0,
                "contexts": []
            }
        
        largest_window = max(window_sizes)
        contexts = context_analysis["contexts_by_window_size"][largest_window]
        
        # Position each context in the latent space
        for context in contexts:
            context_text = context["text"]
            context["position"] = self.latent_space.position_text(context_text)
        
        # Find semantic clusters if there are enough contexts
        clusters = {}
        if len(contexts) >= 3:
            cluster_analysis = self.context_analyzer.find_semantic_clusters(contexts)
            if cluster_analysis["success"]:
                clusters = cluster_analysis["clusters"]
        
        return {
            "term": term,
            "occurrences": len(contexts),
            "contexts": contexts,
            "semantic_clusters": clusters,
            "window_size": largest_window
        }
    
    def compare_term_usage_across_texts(self, texts_dict: Dict[str, str], 
                                       term: str) -> Dict[str, Any]:
        """
        Compare how a term is used across different texts.
        
        Args:
            texts_dict: Dictionary mapping text names to text content
            term: Term to analyze
            
        Returns:
            Dictionary with comparison results
        """
        if not texts_dict or not term:
            return {"term": term, "texts_analyzed": 0}
        
        # Analyze term usage in each text
        usage_by_text = {}
        contexts_by_text = {}
        
        for name, text in texts_dict.items():
            analysis = self.analyze_term_usage(text, term)
            usage_by_text[name] = analysis
            
            if analysis["occurrences"] > 0:
                contexts_by_text[name] = analysis["contexts"]
        
        # Compare contexts between texts
        comparisons = {}
        text_names = list(contexts_by_text.keys())
        
        for i, name1 in enumerate(text_names):
            for name2 in text_names[i+1:]:
                if name1 in contexts_by_text and name2 in contexts_by_text:
                    comparison = self.context_analyzer.compare_contexts(
                        contexts_by_text[name1],
                        contexts_by_text[name2]
                    )
                    comparisons[f"{name1}_vs_{name2}"] = comparison
        
        return {
            "term": term,
            "texts_analyzed": len(texts_dict),
            "texts_with_term": len(contexts_by_text),
            "usage_by_text": usage_by_text,
            "comparisons": comparisons
        }
    
    def find_distinctive_terms(self, text1: str, text2: str, 
                              top_n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find terms that are distinctive to each text compared to the other.
        
        Args:
            text1: First text
            text2: Second text
            top_n: Number of top distinctive terms to extract
            
        Returns:
            Dictionary with distinctive terms for each text
        """
        if not text1 or not text2:
            return {"text1": [], "text2": []}
        
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            stop_words=self._stopwords,
            ngram_range=(1, 3),    # Include single words, bigrams, and trigrams
            # No min_df parameter for small corpus
            max_df=1.0             # Allow terms that appear in all documents
        )
        
        try:
            # Generate TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get scores for each text
            scores1 = tfidf_matrix[0].toarray()[0]
            scores2 = tfidf_matrix[1].toarray()[0]
            
            # Calculate difference in scores
            diff1 = scores1 - scores2  # Terms more important in text1
            diff2 = scores2 - scores1  # Terms more important in text2
            
            # Get distinctive terms for each text
            distinctive1 = [(feature_names[i], diff1[i]) 
                           for i in np.argsort(diff1)[::-1] 
                           if diff1[i] > 0 and len(feature_names[i]) > 2]
            
            distinctive2 = [(feature_names[i], diff2[i]) 
                           for i in np.argsort(diff2)[::-1] 
                           if diff2[i] > 0 and len(feature_names[i]) > 2]
            
            return {
                "text1": distinctive1[:top_n],
                "text2": distinctive2[:top_n]
            }
        
        except Exception as e:
            print(f"Error finding distinctive terms: {e}")
            return {"text1": [], "text2": []}
    
    def _get_stopwords(self) -> List[str]:
        """
        Get a list of German stopwords.
        
        Returns:
            List of stopwords
        """
        # Basic German stopwords
        german_stopwords = [
            "aber", "alle", "allem", "allen", "aller", "alles", "als", "also", "am", "an", "ander", "andere",
            "anderem", "anderen", "anderer", "anderes", "anderm", "andern", "anderr", "anders", "auch", "auf",
            "aus", "bei", "bin", "bis", "bist", "da", "damit", "dann", "der", "den", "des", "dem", "die", "das",
            "daß", "dass", "derselbe", "derselben", "denselben", "desselben", "demselben", "dieselbe", "dieselben",
            "dasselbe", "dazu", "dein", "deine", "deinem", "deinen", "deiner", "deines", "denn", "derer", "dessen",
            "dich", "dir", "du", "dies", "diese", "diesem", "diesen", "dieser", "dieses", "doch", "dort", "durch",
            "ein", "eine", "einem", "einen", "einer", "eines", "einig", "einige", "einigem", "einigen", "einiger",
            "einiges", "einmal", "er", "ihn", "ihm", "es", "etwas", "euer", "eure", "eurem", "euren", "eurer",
            "eures", "für", "gegen", "gewesen", "hab", "habe", "haben", "hat", "hatte", "hatten", "hier", "hin",
            "hinter", "ich", "mich", "mir", "ihr", "ihre", "ihrem", "ihren", "ihrer", "ihres", "euch", "im", "in",
            "indem", "ins", "ist", "jede", "jedem", "jeden", "jeder", "jedes", "jene", "jenem", "jenen", "jener",
            "jenes", "jetzt", "kann", "kein", "keine", "keinem", "keinen", "keiner", "keines", "können", "könnte",
            "machen", "man", "manche", "manchem", "manchen", "mancher", "manches", "mein", "meine", "meinem",
            "meinen", "meiner", "meines", "mit", "muss", "musste", "nach", "nicht", "nichts", "noch", "nun", "nur",
            "ob", "oder", "ohne", "sehr", "sein", "seine", "seinem", "seinen", "seiner", "seines", "selbst", "sich",
            "sie", "ihnen", "sind", "so", "solche", "solchem", "solchen", "solcher", "solches", "soll", "sollte",
            "sondern", "sonst", "über", "um", "und", "uns", "unse", "unsem", "unsen", "unser", "unses", "unter",
            "viel", "vom", "von", "vor", "während", "war", "waren", "warst", "was", "weg", "weil", "weiter",
            "welche", "welchem", "welchen", "welcher", "welches", "wenn", "werde", "werden", "wie", "wieder",
            "will", "wir", "wird", "wirst", "wo", "wollen", "wollte", "würde", "würden", "zu", "zum", "zur", "zwar",
            "zwischen"
        ]
        
        return german_stopwords
