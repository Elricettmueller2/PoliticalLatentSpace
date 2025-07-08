#!/usr/bin/env python
"""
Test and validate the multi-level latent space implementation.

This script loads the multi-level latent space and tests its key features:
1. Entity retrieval and information
2. Word embedding retrieval and similarity search
3. Nearest words to entities
4. Hybrid visualization data generation
5. Word cloud generation

Usage:
    python test_multi_level_latent_space.py --data-dir src/data/processed
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from pprint import pprint
import logging

# Add the project root to the path so we can import our modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.embeddings.multi_level_latent_space import MultiLevelLatentSpace

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_entity_retrieval(latent_space):
    """Test entity retrieval functionality."""
    logger.info("Testing entity retrieval...")
    
    # Get all movements
    movements = list(latent_space.entity_data.get('movements', {}).keys())
    logger.info(f"Found {len(movements)} movements:")
    for movement in movements:
        logger.info(f"  - {movement}")
    
    # Get all politicians
    politicians = list(latent_space.entity_data.get('politicians', {}).keys())
    logger.info(f"Found {len(politicians)} politicians")
    
    # Get entity position for a specific movement
    if movements:
        test_movement = movements[0]
        movement_info = latent_space.get_entity_position('movement', test_movement)
        logger.info(f"\nPosition for movement '{test_movement}':")
        pprint(movement_info)
    
    # Get entity position for a specific politician
    if politicians:
        test_politician = politicians[0]
        politician_info = latent_space.get_entity_position('politician', test_politician)
        logger.info(f"\nPosition for politician '{test_politician}':")
        pprint(politician_info)
    
    # Get entity embeddings
    if movements:
        test_movement = movements[0]
        movement_embedding = latent_space.get_entity_embedding('movement', test_movement)
        if movement_embedding is not None:
            logger.info(f"\nEmbedding for movement '{test_movement}' (shape: {np.array(movement_embedding).shape}):")        
            logger.info(f"  First 5 dimensions: {np.array(movement_embedding)[:5]}")


def test_word_embeddings(latent_space):
    """Test word embedding functionality."""
    logger.info("\nTesting word embeddings...")
    
    # Test words to check
    test_words = ["politik", "wirtschaft", "umwelt", "sozial", "deutschland"]
    
    # Check if word embedding store is available
    if latent_space.word_embedding_store is None:
        logger.warning("Word embedding store is not available. Skipping word embedding tests.")
        return
    
    # Get embedding for a word
    for word in test_words:
        embedding = latent_space.word_embedding_store.get_word_embedding(word)
        if embedding is not None:
            logger.info(f"Embedding for word '{word}' (shape: {embedding.shape}):")
            logger.info(f"  First 5 dimensions: {embedding[:5]}")
            break
    
    # Find similar words
    for word in test_words:
        embedding = latent_space.word_embedding_store.get_word_embedding(word)
        if embedding is not None:
            nearest_words = latent_space.word_embedding_store.get_nearest_words(embedding, top_n=5)
            logger.info(f"\nTop 5 similar words to '{word}':")
            for word_info in nearest_words:
                logger.info(f"  - {word_info['word']}: {word_info['distance']:.4f}")
            break


def test_nearest_words_to_entity(latent_space):
    """Test finding nearest words to entities."""
    logger.info("\nTesting nearest words to entities...")
    
    # Get all movements
    movements = list(latent_space.entity_data.get('movements', {}).keys())
    
    # Find nearest words to a movement
    if movements and latent_space.word_embedding_store is not None:
        test_movement = movements[0]
        nearest_words = latent_space.get_nearest_words_to_entity('movement', test_movement, top_n=10)
        logger.info(f"\nTop 10 words closest to movement '{test_movement}':")
        if nearest_words:
            for i, (word, distance) in enumerate(nearest_words, 1):
                logger.info(f"  {i}. {word} (distance: {distance:.4f})")
        else:
            logger.warning("No nearest words found.")
    
    # Get all politicians
    politicians = list(latent_space.entity_data.get('politicians', {}).keys())
    
    # Find nearest words to a politician
    if politicians and latent_space.word_embedding_store is not None:
        test_politician = politicians[0]
        nearest_words = latent_space.get_nearest_words_to_entity('politician', test_politician, top_n=10)
        logger.info(f"\nTop 10 words closest to politician '{test_politician}':")
        if nearest_words:
            for i, (word, distance) in enumerate(nearest_words, 1):
                logger.info(f"  {i}. {word} (distance: {distance:.4f})")
        else:
            logger.warning("No nearest words found.")


def test_hybrid_visualization(latent_space):
    """Test hybrid visualization data generation."""
    logger.info("\nTesting hybrid visualization data generation...")
    
    # Generate hybrid visualization data
    viz_data = latent_space.get_hybrid_visualization_data(num_words=20)
    
    if viz_data:
        # Print summary of visualization data
        logger.info(f"Generated visualization data with {len(viz_data.get('entities', []))} entities "  
                    f"and {len(viz_data.get('words', []))} words")
        
        # Print sample of entity data
        entities = viz_data.get('entities', {})
        if entities and 'movements' in entities and entities['movements']:
            logger.info("\nSample movement data:")
            pprint(entities['movements'][0])
        if entities and 'politicians' in entities and entities['politicians']:
            logger.info("\nSample politician data:")
            pprint(entities['politicians'][0])
        
        # Print sample of word data
        words = viz_data.get('words', [])
        if words:
            logger.info("\nSample word data:")
            pprint(words[0])
    else:
        logger.warning("No visualization data generated.")


def test_word_cloud(latent_space):
    """Test word cloud generation."""
    logger.info("\nTesting word cloud generation...")
    
    # Get all movements
    movements = list(latent_space.entity_data.get('movements', {}).keys())
    
    # Generate word cloud for a movement
    if movements and latent_space.word_embedding_store is not None:
        test_movement = movements[0]
        word_cloud = latent_space.get_word_cloud_for_entity('movement', test_movement, top_n=20)
        logger.info(f"\nWord cloud for movement '{test_movement}' (top 20 words):")
        if word_cloud:
            for i, (word, weight) in enumerate(word_cloud, 1):
                logger.info(f"  {i}. {word} (weight: {weight:.4f})")
        else:
            logger.warning("No word cloud data generated.")


def main():
    parser = argparse.ArgumentParser(description='Test the multi-level latent space implementation')
    parser.add_argument('--data-dir', type=str, default='src/data/processed',
                        help='Directory containing processed data')
    parser.add_argument('--entity-file', type=str, default='multi_level_analysis_results.json',
                        help='Filename for entity-level data')
    parser.add_argument('--embedding-file', type=str, default='word_embeddings.h5',
                        help='Filename for word embeddings')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Check if the data files exist
    entity_path = os.path.join(args.data_dir, args.entity_file)
    embedding_path = os.path.join(args.data_dir, args.embedding_file)
    index_path = os.path.join(args.data_dir, os.path.splitext(args.embedding_file)[0] + '.index')
    
    if not os.path.exists(entity_path):
        logger.error(f"Entity file not found: {entity_path}")
        logger.error("Please run build_multi_level_latent_space.py first.")
        sys.exit(1)
    
    if not os.path.exists(embedding_path):
        logger.error(f"Embedding file not found: {embedding_path}")
        logger.error("Please run build_multi_level_latent_space.py first.")
        sys.exit(1)
    
    if not os.path.exists(index_path):
        logger.error(f"FAISS index file not found: {index_path}")
        logger.error("Please run build_multi_level_latent_space.py first.")
        sys.exit(1)
    
    # Load the multi-level latent space
    logger.info(f"Loading multi-level latent space from {args.data_dir}...")
    latent_space = MultiLevelLatentSpace(
        entity_data_path=os.path.join(args.data_dir, args.entity_file),
        word_embedding_path=os.path.join(args.data_dir, args.embedding_file),
        index_file=os.path.join(args.data_dir, 'word_embeddings.index'),
        verbose=args.verbose
    )
    
    # Run tests
    test_entity_retrieval(latent_space)
    test_word_embeddings(latent_space)
    test_nearest_words_to_entity(latent_space)
    test_hybrid_visualization(latent_space)
    test_word_cloud(latent_space)
    
    logger.info("\nAll tests completed successfully!")


if __name__ == '__main__':
    main()
