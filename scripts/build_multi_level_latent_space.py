#!/usr/bin/env python
"""
Build a new multi-level latent space from scratch.

This script creates a hybrid latent space that combines:
1. Entity-level embeddings (parties, politicians) - compact, interpretable
2. Word-level embeddings - detailed, rich

The resulting latent space supports both high-level political landscape visualization
and detailed semantic exploration with drill-down capabilities.
"""

import os
import sys
import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import logging

# Add the project root to the path so we can import our modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import PoliticalTextLoader
from src.data.preprocessor import TextPreprocessor
from src.data.embeddings.multi_level_latent_space import MultiLevelLatentSpace

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Political dimension axes
DIMENSIONS = {
    'economic': {'left': -1.0, 'right': 1.0},  # Social Democratic (-) vs. Market Liberal (+)
    'social': {'conservative': -1.0, 'progressive': 1.0},  # Conservative (-) vs. Progressive (+)
    'ecological': {'industrial': -1.0, 'green': 1.0}  # Industrial (-) vs. Green (+)
}

# Expert-defined positions for German political movements
MOVEMENT_POSITIONS = {
    'cdu': {
        'economic': 0.3,    # Center-right
        'social': -0.5,     # Conservative
        'ecological': -0.1  # Slightly industrial
    },
    'spd': {
        'economic': -0.5,   # Center-left
        'social': 0.3,      # Somewhat progressive
        'ecological': 0.2   # Slightly green
    },
    'fdp': {
        'economic': 0.8,    # Market liberal
        'social': 0.5,      # Progressive
        'ecological': -0.2  # Somewhat industrial
    },
    'gruene': {
        'economic': -0.3,   # Somewhat social democratic
        'social': 0.7,      # Progressive
        'ecological': 0.9   # Strongly green
    },
    'linke': {
        'economic': -0.9,   # Strongly social democratic
        'social': 0.6,      # Progressive
        'ecological': 0.4   # Somewhat green
    },
    'afd': {
        'economic': 0.2,    # Slightly market liberal
        'social': -0.9,     # Strongly conservative
        'ecological': -0.7  # Industrial
    },
    'volt': {
        'economic': 0.0,    # Centrist
        'social': 0.8,      # Strongly progressive
        'ecological': 0.7   # Green
    },
    'bsw': {
        'economic': -0.6,   # Social democratic
        'social': 0.0,      # Centrist
        'ecological': 0.1   # Slightly green
    }
}

# Colors for movements in visualization
MOVEMENT_COLORS = {
    'cdu': 'rgba(0, 0, 0, 0.7)',      # Black
    'spd': 'rgba(255, 0, 0, 0.7)',    # Red
    'fdp': 'rgba(255, 255, 0, 0.7)',  # Yellow
    'gruene': 'rgba(0, 128, 0, 0.7)', # Green
    'linke': 'rgba(128, 0, 128, 0.7)', # Purple
    'afd': 'rgba(0, 0, 255, 0.7)',    # Blue
    'volt': 'rgba(128, 0, 255, 0.7)', # Purple-blue
    'bsw': 'rgba(165, 42, 42, 0.7)'   # Brown
}


def load_political_texts(raw_data_dir: str) -> Dict[str, Any]:
    """
    Load political texts from the raw data directory.
    
    Args:
        raw_data_dir: Path to raw data directory
        
    Returns:
        Dictionary with movement and politician texts
    """
    logger.info(f"Loading political texts from {raw_data_dir}")
    
    loader = PoliticalTextLoader(base_dir=raw_data_dir)
    
    # Load movement texts
    movement_texts = loader.load_movement_texts()
    logger.info(f"Loaded texts for {len(movement_texts)} movements")
    
    # Load politician texts
    politician_texts = loader.load_politician_texts()
    logger.info(f"Loaded texts for {len(politician_texts)} politicians")
    
    return {
        'movements': movement_texts,
        'politicians': politician_texts
    }


def preprocess_texts(texts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess the political texts.
    
    Args:
        texts: Dictionary with movement and politician texts
        
    Returns:
        Dictionary with preprocessed texts
    """
    logger.info("Preprocessing texts")
    
    preprocessor = TextPreprocessor(language='german')
    
    # Preprocess movement texts
    movement_texts = preprocessor.preprocess_texts(texts['movements'])
    
    # Preprocess politician texts (more complex structure)
    politician_texts = {}
    for name, data in texts['politicians'].items():
        if isinstance(data, dict) and 'text' in data:
            processed_data = data.copy()
            processed_data['text'] = preprocessor.preprocess_text(data['text'])
            politician_texts[name] = processed_data
    
    return {
        'movements': movement_texts,
        'politicians': politician_texts
    }


def generate_embeddings(texts: Dict[str, Any], embedding_dim: int = 300) -> Dict[str, Any]:
    """
    Generate embeddings for texts and words.
    
    In a real implementation, this would use a pre-trained model like Word2Vec, 
    FastText, or a transformer model. For this example, we'll generate random 
    embeddings to demonstrate the structure.
    
    Args:
        texts: Dictionary with preprocessed texts
        embedding_dim: Dimension of embeddings
        
    Returns:
        Dictionary with entity embeddings and word embeddings
    """
    logger.info(f"Generating embeddings with dimension {embedding_dim}")
    
    # Extract all words from texts
    all_words = set()
    
    # Process movement texts
    for movement, text in texts['movements'].items():
        words = text.lower().split()
        all_words.update(words)
    
    # Process politician texts
    for politician, data in texts['politicians'].items():
        if isinstance(data, dict) and 'text' in data:
            words = data['text'].lower().split()
            all_words.update(words)
    
    # Generate word embeddings (in a real implementation, this would use a pre-trained model)
    word_embeddings = {}
    for word in tqdm(all_words, desc="Generating word embeddings"):
        # Generate a random embedding vector
        # In a real implementation, this would be from a model like Word2Vec or FastText
        word_embeddings[word] = np.random.normal(0, 0.1, embedding_dim).astype(np.float32)
    
    logger.info(f"Generated embeddings for {len(word_embeddings)} words")
    
    # Generate entity embeddings by averaging word embeddings
    entity_embeddings = {
        'movements': {},
        'politicians': {}
    }
    
    # Generate movement embeddings
    for movement, text in texts['movements'].items():
        words = text.lower().split()
        if words:
            # Filter to words that have embeddings
            valid_words = [w for w in words if w in word_embeddings]
            if valid_words:
                # Average the embeddings of all words in the text
                embedding = np.mean([word_embeddings[w] for w in valid_words], axis=0)
                entity_embeddings['movements'][movement] = embedding.tolist()
    
    # Generate politician embeddings
    for politician, data in texts['politicians'].items():
        if isinstance(data, dict) and 'text' in data:
            words = data['text'].lower().split()
            if words:
                # Filter to words that have embeddings
                valid_words = [w for w in words if w in word_embeddings]
                if valid_words:
                    # Average the embeddings of all words in the text
                    embedding = np.mean([word_embeddings[w] for w in valid_words], axis=0)
                    entity_embeddings['politicians'][politician] = embedding.tolist()
    
    logger.info(f"Generated embeddings for {len(entity_embeddings['movements'])} movements "
                f"and {len(entity_embeddings['politicians'])} politicians")
    
    return {
        'entity_embeddings': entity_embeddings,
        'word_embeddings': word_embeddings
    }


def create_entity_data(texts: Dict[str, Any], embeddings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create entity-level data for the multi-level latent space.
    
    Args:
        texts: Dictionary with preprocessed texts
        embeddings: Dictionary with entity embeddings
        
    Returns:
        Entity data dictionary
    """
    logger.info("Creating entity-level data")
    
    entity_data = {
        'movements': {},
        'politicians': {}
    }
    
    # Process movements
    for movement, text in texts['movements'].items():
        # Skip if no embedding
        if movement not in embeddings['entity_embeddings']['movements']:
            continue
        
        # Get expert-defined position
        position = MOVEMENT_POSITIONS.get(movement.lower(), {
            'economic': 0.0,
            'social': 0.0,
            'ecological': 0.0
        })
        
        # Calculate 2D coordinates from the position
        # This is a simple projection, but could be more sophisticated
        x = position['economic']  # Economic axis
        y = position['social']    # Social axis
        
        # Create movement data
        entity_data['movements'][movement] = {
            'text': text,
            'embedding': embeddings['entity_embeddings']['movements'][movement],
            'position': {
                'coordinates': {'x': x, 'y': y},
                'size': 15,  # Larger size for movements
                'color': MOVEMENT_COLORS.get(movement.lower(), 'rgba(128, 128, 128, 0.7)'),
                'expert_dimensions': {
                    'axes': {
                        'economic': position['economic'],
                        'social': position['social'],
                        'ecological': position['ecological']
                    }
                }
            }
        }
    
    # Process politicians
    for politician, data in texts['politicians'].items():
        # Skip if no embedding or not a dict
        if not isinstance(data, dict) or politician not in embeddings['entity_embeddings']['politicians']:
            continue
        
        movement = data.get('movement', '').lower()
        
        # Get position based on movement with some random variation
        base_position = MOVEMENT_POSITIONS.get(movement, {
            'economic': 0.0,
            'social': 0.0,
            'ecological': 0.0
        })
        
        # Add some random variation to make politicians different from their movement
        position = {
            'economic': base_position['economic'] + np.random.normal(0, 0.1),
            'social': base_position['social'] + np.random.normal(0, 0.1),
            'ecological': base_position['ecological'] + np.random.normal(0, 0.1)
        }
        
        # Ensure values are within bounds
        for dim in position:
            position[dim] = max(-1.0, min(1.0, position[dim]))
        
        # Calculate 2D coordinates
        x = position['economic']  # Economic axis
        y = position['social']    # Social axis
        
        # Create politician data
        entity_data['politicians'][politician] = {
            'text': data['text'],
            'movement': movement,
            'embedding': embeddings['entity_embeddings']['politicians'][politician],
            'position': {
                'coordinates': {'x': x, 'y': y},
                'size': 8,  # Smaller size for politicians
                'color': MOVEMENT_COLORS.get(movement, 'rgba(128, 128, 128, 0.7)'),
                'expert_dimensions': {
                    'axes': {
                        'economic': position['economic'],
                        'social': position['social'],
                        'ecological': position['ecological']
                    }
                }
            }
        }
    
    logger.info(f"Created data for {len(entity_data['movements'])} movements "
                f"and {len(entity_data['politicians'])} politicians")
    
    return entity_data


def main():
    parser = argparse.ArgumentParser(description='Build a multi-level political latent space')
    parser.add_argument('--raw-dir', type=str, default='src/data/raw',
                        help='Directory containing raw political texts')
    parser.add_argument('--output-dir', type=str, default='src/data/processed',
                        help='Directory to save processed data')
    parser.add_argument('--entity-file', type=str, default='multi_level_analysis_results.json',
                        help='Filename for entity-level data')
    parser.add_argument('--embedding-file', type=str, default='word_embeddings.h5',
                        help='Filename for word embeddings')
    parser.add_argument('--embedding-dim', type=int, default=300,
                        help='Dimension of embeddings')
    parser.add_argument('--index-type', type=str, default='flat', choices=['flat', 'ivf', 'hnsw'],
                        help='Type of FAISS index to create')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and preprocess texts
    raw_texts = load_political_texts(args.raw_dir)
    processed_texts = preprocess_texts(raw_texts)
    
    # Generate embeddings
    embeddings = generate_embeddings(processed_texts, embedding_dim=args.embedding_dim)
    
    # Create entity data
    entity_data = create_entity_data(processed_texts, embeddings)
    
    # Create multi-level latent space
    latent_space = MultiLevelLatentSpace(verbose=args.verbose)
    
    # Save the latent space
    success = latent_space.create_new_latent_space(
        entity_data=entity_data,
        word_embeddings=embeddings['word_embeddings'],
        output_dir=args.output_dir,
        entity_output_file=args.entity_file,
        word_embedding_output_file=args.embedding_file,
        index_type=args.index_type
    )
    
    if success:
        logger.info(f"Successfully created multi-level latent space in {args.output_dir}")
        logger.info(f"Entity data: {os.path.join(args.output_dir, args.entity_file)}")
        logger.info(f"Word embeddings: {os.path.join(args.output_dir, args.embedding_file)}")
        logger.info(f"FAISS index: {os.path.join(args.output_dir, os.path.splitext(args.embedding_file)[0] + '.index')}")
    else:
        logger.error("Failed to create multi-level latent space")
        sys.exit(1)


if __name__ == '__main__':
    main()
