"""
Text preprocessing functionality for political texts.
"""

import re
from typing import Dict, List, Any, Optional, Callable
import nltk
from nltk.tokenize import sent_tokenize


class TextPreprocessor:
    """
    Preprocesses political texts for analysis.
    """
    
    def __init__(self, language='german'):
        """
        Initialize the text preprocessor.
        
        Args:
            language: Language of the texts
        """
        self.language = language
        self._ensure_nltk_resources()
    
    def preprocess_texts(self, texts_dict: Dict[str, str]) -> Dict[str, str]:
        """
        Preprocess a dictionary of texts.
        
        Args:
            texts_dict: Dictionary mapping names to raw text content
            
        Returns:
            Dictionary mapping names to preprocessed text content
        """
        return {name: self.preprocess_text(text) for name, text in texts_dict.items()}
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a single text.
        
        Args:
            text: Raw text content
            
        Returns:
            Preprocessed text content
        """
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)\[\]\{\}\"\'äöüÄÖÜß]', ' ', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Trim leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def segment_text(self, text: str, max_length: int = 1000, 
                    overlap: int = 100) -> List[str]:
        """
        Segment text into smaller chunks for processing.
        
        Args:
            text: Text to segment
            max_length: Maximum length of each segment
            overlap: Overlap between segments
            
        Returns:
            List of text segments
        """
        if not text or len(text) <= max_length:
            return [text] if text else []
        
        # Try to segment at sentence boundaries
        try:
            sentences = sent_tokenize(text, language=self._get_nltk_language())
            
            segments = []
            current_segment = ""
            
            for sentence in sentences:
                # If adding this sentence would exceed max_length, save current segment
                if len(current_segment) + len(sentence) > max_length and current_segment:
                    segments.append(current_segment.strip())
                    # Keep some overlap by including the last few sentences again
                    overlap_point = self._find_overlap_point(current_segment, overlap)
                    current_segment = current_segment[overlap_point:] if overlap_point > 0 else ""
                
                current_segment += " " + sentence
            
            # Add the last segment if it's not empty
            if current_segment.strip():
                segments.append(current_segment.strip())
            
            return segments
            
        except Exception as e:
            print(f"Error in sentence tokenization: {e}")
            # Fall back to character-based segmentation
            return self._segment_by_chars(text, max_length, overlap)
    
    def _segment_by_chars(self, text: str, max_length: int = 1000, 
                         overlap: int = 100) -> List[str]:
        """
        Segment text by characters when sentence tokenization fails.
        
        Args:
            text: Text to segment
            max_length: Maximum length of each segment
            overlap: Overlap between segments
            
        Returns:
            List of text segments
        """
        segments = []
        start = 0
        
        while start < len(text):
            # Find a good breaking point (end of sentence or paragraph)
            end = min(start + max_length, len(text))
            
            if end < len(text):
                # Look for sentence boundaries (., !, ?)
                sentence_end = max(
                    text.rfind('. ', start, end),
                    text.rfind('! ', start, end),
                    text.rfind('? ', start, end)
                )
                
                if sentence_end > start + max_length // 2:
                    end = sentence_end + 1  # Include the period
                else:
                    # If no good sentence boundary, look for space
                    space = text.rfind(' ', start + max_length // 2, end)
                    if space > start:
                        end = space
            
            segments.append(text[start:end].strip())
            start = max(start + 1, end - overlap)  # Ensure we make progress
        
        return segments
    
    def _find_overlap_point(self, text: str, overlap_chars: int) -> int:
        """
        Find a good point for overlap (preferably at sentence boundary).
        
        Args:
            text: Text to find overlap point in
            overlap_chars: Approximate number of characters to overlap
            
        Returns:
            Index where overlap should start
        """
        if len(text) <= overlap_chars:
            return 0
        
        # Look for sentence boundaries in the overlap region
        start_idx = max(0, len(text) - overlap_chars * 2)
        overlap_text = text[start_idx:]
        
        # Find the last sentence boundary
        sentence_end = max(
            overlap_text.rfind('. '),
            overlap_text.rfind('! '),
            overlap_text.rfind('? ')
        )
        
        if sentence_end > 0:
            return start_idx + sentence_end + 2  # Include the period and space
        
        # If no sentence boundary, find a space
        space = overlap_text.rfind(' ', overlap_chars // 2)
        if space > 0:
            return start_idx + space + 1  # Include the space
        
        # If all else fails, just use character-based overlap
        return max(0, len(text) - overlap_chars)
    
    def clean_and_normalize(self, text: str) -> str:
        """
        Clean and normalize text (more aggressive than preprocess_text).
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, emails, and special characters
        text = self.preprocess_text(text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Replace German umlauts
        text = text.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
        
        # Remove all non-alphanumeric characters
        text = re.sub(r'[^a-z ]', ' ', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def apply_custom_preprocessing(self, texts_dict: Dict[str, str], 
                                 preprocessor: Callable[[str], str]) -> Dict[str, str]:
        """
        Apply a custom preprocessing function to texts.
        
        Args:
            texts_dict: Dictionary mapping names to text content
            preprocessor: Custom preprocessing function
            
        Returns:
            Dictionary with preprocessed texts
        """
        return {name: preprocessor(text) for name, text in texts_dict.items()}
    
    def _ensure_nltk_resources(self):
        """
        Ensure required NLTK resources are downloaded.
        """
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
            except Exception as e:
                print(f"Warning: Could not download NLTK resources: {e}")
                print("Falling back to simple tokenization.")
    
    def _get_nltk_language(self) -> str:
        """
        Get the NLTK language code for the current language.
        
        Returns:
            NLTK language code
        """
        language_map = {
            'german': 'german',
            'deutsch': 'german',
            'english': 'english',
            'englisch': 'english'
        }
        
        return language_map.get(self.language.lower(), 'german')
