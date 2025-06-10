"""
Enhanced embedding functionality for political text analysis.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Dict, Any
import re


class EnhancedEmbedder:
    """
    Enhanced embedding functionality with support for efficient batch processing
    and text segmentation for long documents.
    """
    
    def __init__(self, model_name='distiluse-base-multilingual-cased-v2', batch_size=32):
        """
        Initialize the embedder with a specific model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            batch_size: Batch size for efficient processing
        """
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.model_name = model_name
    
    def encode(self, text: str, segment: bool = False, segment_size: int = 512, 
               overlap: int = 50) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for text, optionally segmenting long texts.
        
        Args:
            text: The text to encode
            segment: Whether to segment the text into smaller chunks
            segment_size: Maximum size of each segment in characters
            overlap: Overlap between segments in characters
            
        Returns:
            Either a single embedding vector or a list of embedding vectors if segment=True
        """
        if not segment:
            return self.model.encode(text)
        
        # Segment text and encode in batches
        segments = self._segment_text(text, segment_size, overlap)
        if not segments:
            return np.zeros(self.model.get_sentence_embedding_dimension())
        
        embeddings = self.model.encode(segments, batch_size=self.batch_size)
        return embeddings
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Efficiently encode a batch of texts.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of embeddings
        """
        if not texts:
            return np.array([])
        return self.model.encode(texts, batch_size=self.batch_size)
    
    def _segment_text(self, text: str, segment_size: int = 512, 
                     overlap: int = 50) -> List[str]:
        """
        Segment text into smaller chunks with overlap.
        
        Args:
            text: Text to segment
            segment_size: Maximum size of each segment
            overlap: Overlap between segments
            
        Returns:
            List of text segments
        """
        if not text or len(text) <= segment_size:
            return [text] if text else []
        
        segments = []
        start = 0
        
        while start < len(text):
            # Find a good breaking point (end of sentence or paragraph)
            end = min(start + segment_size, len(text))
            
            # Try to find a sentence boundary
            if end < len(text):
                # Look for sentence boundaries (., !, ?)
                sentence_boundaries = [m.end() for m in re.finditer(r'[.!?]\s+', text[start:end])]
                
                if sentence_boundaries:
                    # Use the last sentence boundary
                    end = start + sentence_boundaries[-1]
                else:
                    # If no sentence boundary, try to find a space
                    spaces = [m.start() for m in re.finditer(r'\s', text[end-20:end])]
                    if spaces:
                        end = end - 20 + spaces[-1]
            
            segments.append(text[start:end])
            start = max(start + 1, end - overlap)  # Ensure we make progress
        
        return segments
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current embedding model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "name": self.model_name,
            "dimension": self.model.get_sentence_embedding_dimension(),
            "batch_size": self.batch_size
        }
