#!/usr/bin/env python3
"""
Latent Space Analysis Tool

This script provides utilities to directly work with the Political Latent Space,
focusing on analyzing word clouds and their relationships to political entities.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional, Set

# Add the project root to the path so we can import the modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the project modules
from src.data.embeddings.multi_level_latent_space import MultiLevelLatentSpace
from src.data.embeddings.chunked_embedding_store import ChunkedEmbeddingStore

class LatentSpaceAnalyzer:
    """
    A class for analyzing the Political Latent Space, with a focus on word clouds
    and their relationships to political entities.
    """
    
    # Common German stopwords/filler words to filter out
    GERMAN_STOPWORDS = {
        # Articles
        'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einer', 'eines', 'einem', 'einen',
        
        # Pronouns
        'ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr', 'sie', 'mich', 'mein', 'mir', 'dich', 'dein', 'dir',
        'sein', 'ihn', 'ihm', 'uns', 'unser', 'euch', 'euer', 'ihnen', 'ihrer',
        
        # Prepositions
        'in', 'an', 'auf', 'für', 'von', 'mit', 'bei', 'nach', 'aus', 'zu', 'um', 'über', 'unter', 'vor',
        'hinter', 'neben', 'zwischen', 'durch', 'gegen', 'ohne', 'bis', 'entlang',
        
        # Conjunctions
        'und', 'oder', 'aber', 'denn', 'sondern', 'als', 'wie', 'wenn', 'weil', 'dass', 'ob',
        
        # Adverbs
        'hier', 'dort', 'dann', 'heute', 'jetzt', 'immer', 'nie', 'so', 'auch', 'nur', 'noch', 'schon',
        'sehr', 'mehr', 'wieder', 'bereits', 'oft', 'manchmal',
        
        # Auxiliary verbs
        'sein', 'haben', 'werden', 'können', 'müssen', 'sollen', 'wollen', 'dürfen', 'mögen',
        'ist', 'sind', 'war', 'waren', 'hat', 'hatte', 'hatten', 'wird', 'werden', 'wurde', 'wurden',
        'kann', 'können', 'konnte', 'konnten', 'muss', 'müssen', 'musste', 'mussten',
        
        # Numbers
        'eins', 'zwei', 'drei', 'vier', 'fünf', 'sechs', 'sieben', 'acht', 'neun', 'zehn',
        
        # Other common filler words
        'ja', 'nein', 'nicht', 'kein', 'keine', 'mal', 'man', 'was', 'wer', 'wo', 'wie', 'warum',
        'weshalb', 'welche', 'welcher', 'welches'
    }
    
    def __init__(self, entity_data_path=None, word_embedding_path=None, index_file=None):
        """
        Initialize the LatentSpaceAnalyzer with paths to the necessary data files.
        
        Args:
            entity_data_path: Path to the entity data JSON file
            word_embedding_path: Path to the word embeddings HDF5 file
            index_file: Path to the FAISS index file
        """
        # Default paths
        if entity_data_path is None:
            entity_data_path = os.path.join(project_root, 'src/data/processed/multi_level_analysis_results.json')
        if word_embedding_path is None:
            word_embedding_path = os.path.join(project_root, 'src/data/processed/word_embeddings.h5')
        if index_file is None:
            index_file = os.path.join(project_root, 'src/data/processed/word_embeddings.index')
        
        # Initialize the latent space
        print(f"Loading latent space from {entity_data_path} and {word_embedding_path}...")
        self.latent_space = MultiLevelLatentSpace(
            entity_data_path=entity_data_path,
            word_embedding_path=word_embedding_path,
            index_file=index_file,
            verbose=True
        )
        
        # Load entity data
        with open(entity_data_path, 'r', encoding='utf-8') as f:
            self.entity_data = json.load(f)
        
        # Get lists of entities
        self.movements = list(self.entity_data.get('movements', {}).keys())
        self.politicians = list(self.entity_data.get('politicians', {}).keys())
        
        print(f"Loaded {len(self.movements)} movements and {len(self.politicians)} politicians")
    
    def get_word_cloud(self, entity_type, entity_name, top_n=100, max_distance=5.0, filter_stopwords=True):
        """
        Get the word cloud for a specific entity.
        
        Args:
            entity_type: Type of entity ('movement' or 'politician')
            entity_name: Name of the entity
            top_n: Number of words to include
            max_distance: Maximum distance threshold 
            filter_stopwords: Whether to filter out common German stopwords
            
        Returns:
            List of word cloud items with text, value, and distance
        """
        # Get entity embedding
        entity_embedding = self.latent_space.get_entity_embedding(entity_type, entity_name)
        if entity_embedding is None:
            print(f"Warning: No embedding found for {entity_type} {entity_name}")
            return []
        
        # Try to get nearest words directly from the embedding store
        if self.latent_space.word_embedding_store:
            # Normalize the entity embedding to have the same magnitude as word embeddings
            entity_norm = np.linalg.norm(entity_embedding)
            if entity_norm > 0:
                # Scale up to have norm around 1.7 (similar to word embeddings)
                entity_embedding = entity_embedding * (1.7 / entity_norm)
            
            nearest_words = self.latent_space.word_embedding_store.get_nearest_words(
                entity_embedding,
                top_n=top_n,
                max_distance=max_distance
            )
            
            # Format for word cloud
            word_cloud_data = []
            for w in nearest_words:
                word = w['word']
                
                # Skip stopwords if filtering is enabled
                if filter_stopwords and word.lower() in self.GERMAN_STOPWORDS:
                    continue
                    
                # Calculate similarity in a way that ensures positive values
                # Map distance to a 0-1 range where closer words have higher values
                similarity = max(0.0, 1.0 - (w['distance'] / max_distance))
                
                word_cloud_data.append({
                    'text': word,
                    'value': similarity,
                    'distance': w['distance']
                })
                
            # Sort by similarity (highest first)
            word_cloud_data = sorted(word_cloud_data, key=lambda x: x['value'], reverse=True)
            
            # Limit to top_n words after filtering
            word_cloud_data = word_cloud_data[:top_n]

            # --- Value Normalization ---
            # Rescale the 'value' to a more visually intuitive range (e.g., 0.1 to 1.0)
            if word_cloud_data:
                values = [w['value'] for w in word_cloud_data]
                min_val, max_val = min(values), max(values)
                
                if max_val > min_val:
                    # Rescale to a 0.1 - 1.0 range
                    for w in word_cloud_data:
                        w['value'] = 0.1 + 0.9 * (w['value'] - min_val) / (max_val - min_val)
                else: # All values are the same
                    for w in word_cloud_data:
                        w['value'] = 0.5 # Assign a medium value

            # If we got results, return them
            if word_cloud_data:
                return word_cloud_data
        
        # Fallback: Generate synthetic word cloud based on entity type and political spectrum
        print(f"Using fallback word cloud for {entity_type} {entity_name}")
        
        # Get entity data
        if entity_type == 'movement':
            entity_data = self.entity_data['movements'].get(entity_name, {})
        else:
            entity_data = self.entity_data['politicians'].get(entity_name, {})
        
        # Get political spectrum
        spectrum = entity_data.get('political_spectrum', 'center')
        
        # Define word sets based on political spectrum
        word_sets = {
            'far-left': ['socialism', 'revolution', 'equality', 'collective', 'workers', 'class', 'solidarity', 'anticapitalism'],
            'left': ['social', 'welfare', 'public', 'redistribution', 'justice', 'rights', 'progressive'],
            'center-left': ['reform', 'progress', 'diversity', 'inclusion', 'environment', 'education', 'healthcare'],
            'center': ['compromise', 'moderate', 'pragmatic', 'balance', 'stability', 'dialogue', 'cooperation'],
            'center-right': ['market', 'economy', 'tradition', 'security', 'responsibility', 'family', 'business'],
            'right': ['conservative', 'nation', 'values', 'order', 'freedom', 'competition', 'individual'],
            'far-right': ['nationalism', 'identity', 'sovereignty', 'borders', 'tradition', 'authority', 'patriotism']
        }
        
        # Get words for this spectrum
        words = word_sets.get(spectrum, word_sets['center'])
        
        # Add some common political terms
        common_words = ['politik', 'demokratie', 'partei', 'wahl', 'bürger', 'regierung', 'parlament']
        words.extend(common_words)
        
        # Add entity-specific words
        if entity_type == 'movement':
            if 'cdu' in entity_name.lower():
                words.extend(['christlich', 'union', 'wirtschaft', 'sicherheit'])
            elif 'spd' in entity_name.lower():
                words.extend(['sozialdemokratie', 'arbeit', 'gerechtigkeit'])
            elif 'gruen' in entity_name.lower() or 'grün' in entity_name.lower():
                words.extend(['umwelt', 'klima', 'nachhaltigkeit', 'ökologie'])
            elif 'fdp' in entity_name.lower():
                words.extend(['freiheit', 'markt', 'liberal', 'wirtschaft'])
            elif 'linke' in entity_name.lower():
                words.extend(['sozialismus', 'gerechtigkeit', 'umverteilung'])
            elif 'afd' in entity_name.lower():
                words.extend(['heimat', 'migration', 'tradition', 'euro'])
        
        # Create synthetic word cloud data
        word_cloud_data = []
        for i, word in enumerate(words):
            # Generate synthetic similarity values
            similarity = 0.9 - (i / len(words) * 0.5)  # Values from 0.9 to 0.4
            word_cloud_data.append({
                'text': word,
                'value': similarity,
                'distance': 1.0 - similarity
            })
        
        return word_cloud_data
    
    def compare_word_clouds(self, entity1, entity2, top_n=100, max_distance=5.0):
        """
        Compare word clouds between two entities.
        
        Args:
            entity1: Tuple of (entity_type, entity_name)
            entity2: Tuple of (entity_type, entity_name)
            top_n: Number of words to include for each entity
            max_distance: Maximum distance threshold
            
        Returns:
            Dictionary with common words, unique words for each entity, and similarity score
        """
        entity1_type, entity1_name = entity1
        entity2_type, entity2_name = entity2
        
        # Map entity names to the correct format if needed
        # This ensures we're using the correct entity names that match the embeddings
        entity_map = {
            'CDU': 'cdu',
            'SPD': 'spd',
            'FDP': 'fdp',
            'AfD': 'afd',
            'CSU': 'csu',
            'Die Linke': 'linke',
            'Bündnis 90/Die Grünen': 'grüne',
            'Gruene': 'grüne',
            'Grüne': 'grüne'
        }
        
        # Apply mapping if available
        if entity1_name in entity_map:
            entity1_name = entity_map[entity1_name]
        elif entity1_name.lower() in entity_map:
            entity1_name = entity_map[entity1_name.lower()]
            
        if entity2_name in entity_map:
            entity2_name = entity_map[entity2_name]
        elif entity2_name.lower() in entity_map:
            entity2_name = entity_map[entity2_name.lower()]
        
        # Get word clouds
        cloud1 = self.get_word_cloud(entity1_type, entity1_name, top_n, max_distance)
        cloud2 = self.get_word_cloud(entity2_type, entity2_name, top_n, max_distance)
        
        # Extract words and values
        words1 = {item['text']: item['value'] for item in cloud1}
        words2 = {item['text']: item['value'] for item in cloud2}
        
        # Find common and unique words
        common_words = set(words1.keys()) & set(words2.keys())
        unique_to_1 = set(words1.keys()) - set(words2.keys())
        unique_to_2 = set(words2.keys()) - set(words1.keys())
        
        # Calculate similarity score (Jaccard similarity of the word sets)
        # Handle empty sets to avoid division by zero
        total_unique_words = len(words1) + len(words2) - len(common_words)
        if total_unique_words > 0:
            similarity = len(common_words) / total_unique_words
        else:
            similarity = 0.0 if len(words1) == 0 and len(words2) == 0 else 1.0
        
        # Create detailed comparison for common words
        common_word_details = [
            {
                'text': word,
                'value1': words1[word],
                'value2': words2[word],
                'difference': words1[word] - words2[word]
            }
            for word in common_words
        ]
        
        # Sort by absolute difference
        common_word_details.sort(key=lambda x: abs(x['difference']), reverse=True)
        
        return {
            'entity1': {'type': entity1_type, 'name': entity1_name},
            'entity2': {'type': entity2_type, 'name': entity2_name},
            'common_words': common_word_details,
            'unique_to_entity1': [{'text': word, 'value': words1[word]} for word in unique_to_1],
            'unique_to_entity2': [{'text': word, 'value': words2[word]} for word in unique_to_2],
            'similarity_score': similarity
        }
    
    def generate_word_cloud_matrix(self, entity_type='movement', top_n=50, max_distance=5.0):
        """
        Generate a similarity matrix between entities based on their word clouds.
        
        Args:
            entity_type: Type of entities to compare ('movement' or 'politician')
            top_n: Number of words to include for each entity
            max_distance: Maximum distance threshold (default: 5.0 to match original implementation)
            
        Returns:
            Pandas DataFrame with similarity scores
        """
        entities = self.movements if entity_type == 'movement' else self.politicians
        
        # Initialize similarity matrix
        similarity_matrix = pd.DataFrame(index=entities, columns=entities)
        
        # Calculate similarity for each pair
        for i, entity1 in enumerate(entities):
            print(f"Processing {i+1}/{len(entities)}: {entity1}")
            for entity2 in entities:
                if entity1 == entity2:
                    similarity_matrix.loc[entity1, entity2] = 1.0
                else:
                    comparison = self.compare_word_clouds(
                        (entity_type, entity1),
                        (entity_type, entity2),
                        top_n=top_n,
                        max_distance=max_distance
                    )
                    similarity_matrix.loc[entity1, entity2] = comparison['similarity_score']
        
        return similarity_matrix
    
    def find_distinctive_words(self, entity_type, entity_name, compare_to=None, top_n=50, max_distance=0.5):
        """
        Find words that are distinctive for an entity compared to others.
        
        Args:
            entity_type: Type of entity ('movement' or 'politician')
            entity_name: Name of the entity
            compare_to: List of entities to compare against (if None, compare to all of same type)
            top_n: Number of words to include
            max_distance: Maximum distance threshold
            
        Returns:
            List of distinctive words with scores
        """
        # Get the entity's word cloud
        entity_cloud = self.get_word_cloud(entity_type, entity_name, top_n=top_n*2, max_distance=max_distance)
        entity_words = {item['text']: item['value'] for item in entity_cloud}
        
        # Determine comparison entities
        if compare_to is None:
            if entity_type == 'movement':
                compare_to = [(entity_type, e) for e in self.movements if e != entity_name]
            else:
                compare_to = [(entity_type, e) for e in self.politicians if e != entity_name]
        
        # Get word clouds for comparison entities
        comparison_clouds = []
        for comp_type, comp_name in compare_to:
            cloud = self.get_word_cloud(comp_type, comp_name, top_n=top_n*2, max_distance=max_distance)
            comparison_clouds.append({item['text']: item['value'] for item in cloud})
        
        # Calculate distinctiveness score for each word
        distinctive_words = []
        for word, value in entity_words.items():
            # Count how many comparison entities have this word
            presence_count = sum(1 for cloud in comparison_clouds if word in cloud)
            
            # Calculate average value in comparison entities
            if presence_count > 0:
                avg_comp_value = sum(cloud.get(word, 0) for cloud in comparison_clouds) / presence_count
            else:
                avg_comp_value = 0
            
            # Distinctiveness score: entity's value minus average comparison value
            distinctiveness = value - avg_comp_value
            
            distinctive_words.append({
                'text': word,
                'entity_value': value,
                'avg_comparison_value': avg_comp_value,
                'distinctiveness': distinctiveness,
                'presence_ratio': presence_count / len(comparison_clouds)
            })
        
        # Sort by distinctiveness
        distinctive_words.sort(key=lambda x: x['distinctiveness'], reverse=True)
        
        return distinctive_words[:top_n]
    
    def plot_word_cloud_similarity(self, similarity_matrix, title=None, figsize=(10, 8)):
        """
        Plot a heatmap of the word cloud similarity matrix.
        
        Args:
            similarity_matrix: Pandas DataFrame with similarity scores
            title: Title for the plot
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=figsize)
        sns.heatmap(similarity_matrix, annot=True, cmap='viridis', vmin=0, vmax=1)
        
        if title:
            plt.title(title)
        else:
            plt.title('Word Cloud Similarity Matrix')
        
        plt.tight_layout()
        return plt.gcf()
    
    def analyze_political_dimensions(self, entity_type, entity_name, top_n=50):
        """
        Analyze the political dimensions of an entity based on its word cloud.
        
        Args:
            entity_type: Type of entity ('movement' or 'politician')
            entity_name: Name of the entity
            top_n: Number of words to include
            
        Returns:
            Dictionary with political dimension scores
        """
        # Get the entity's word cloud
        word_cloud = self.get_word_cloud(entity_type, entity_name, top_n=top_n)
        
        # Political dimension anchors (from app.py)
        # Economic: Market Liberal (+) vs. Social Democratic (-)
        # Social: Progressive (+) vs. Conservative (-)
        # Ecological: Green (+) vs. Industrial (-)
        dimension_words = {
            'economic_liberal': ['market', 'economy', 'taxes', 'business', 'growth', 'investment', 'trade', 'competition'],
            'economic_social': ['welfare', 'redistribution', 'public', 'social', 'services', 'solidarity', 'equality'],
            'social_progressive': ['equality', 'justice', 'rights', 'diversity', 'inclusion', 'reform', 'progress'],
            'social_conservative': ['tradition', 'family', 'security', 'identity', 'sovereignty', 'nation', 'borders'],
            'ecological_green': ['climate', 'environment', 'sustainability', 'renewable', 'conservation', 'emissions'],
            'ecological_industrial': ['industry', 'production', 'jobs', 'manufacturing', 'infrastructure', 'energy']
        }
        
        # Calculate dimension scores
        dimension_scores = {dim: 0 for dim in dimension_words}
        word_contributions = {dim: {} for dim in dimension_words}
        
        for item in word_cloud:
            word = item['text']
            value = item['value']
            
            for dim, words in dimension_words.items():
                if word.lower() in words:
                    dimension_scores[dim] += value
                    word_contributions[dim][word] = value
        
        # Normalize scores
        for dim in dimension_scores:
            if dimension_words[dim]:  # Avoid division by zero
                dimension_scores[dim] /= len(dimension_words[dim])
        
        # Calculate composite scores
        economic_axis = dimension_scores['economic_liberal'] - dimension_scores['economic_social']
        social_axis = dimension_scores['social_progressive'] - dimension_scores['social_conservative']
        ecological_axis = dimension_scores['ecological_green'] - dimension_scores['ecological_industrial']
        
        return {
            'entity': {'type': entity_type, 'name': entity_name},
            'dimension_scores': dimension_scores,
            'composite_scores': {
                'economic_axis': economic_axis,
                'social_axis': social_axis,
                'ecological_axis': ecological_axis
            },
            'word_contributions': word_contributions
        }

def main():
    """Main function to demonstrate the LatentSpaceAnalyzer."""
    # Initialize analyzer
    analyzer = LatentSpaceAnalyzer()
    
    # Example 1: Get word cloud for a specific entity
    print("\n=== Example 1: Word Cloud for CDU ===")
    word_cloud = analyzer.get_word_cloud('movement', 'cdu')
    print(json.dumps(word_cloud[:15], indent=2, ensure_ascii=False))
    
    # Example 2: Compare word clouds between two movements
    print("\n=== Example 2: Comparing CDU and SPD ===")
    comparison = analyzer.compare_word_clouds(('movement', 'cdu'), ('movement', 'spd'))
    print(f"Similarity score: {comparison['similarity_score']:.2f}\n")
    print("Top 5 common words with largest difference:")
    for word in comparison['common_words'][:5]:
        print(f"  {word['text']}: CDU ({word['value1']:.2f}) vs SPD ({word['value2']:.2f}), diff: {word['difference']:.2f}")
    
    # Skip Example 3 and 4 as those methods aren't implemented yet

if __name__ == "__main__":
    main()
