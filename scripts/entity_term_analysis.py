#!/usr/bin/env python3
"""
Entity-Term Analysis Tool

This script analyzes the relationships between political entities and terms
in the latent space, allowing for comparative analysis and visualization.
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
from collections import Counter, defaultdict
import argparse

# Add the project root to the path so we can import the modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our analyzer
from scripts.analyze_latent_space import LatentSpaceAnalyzer

class EntityTermAnalyzer:
    """
    A class for analyzing the relationships between political entities and terms.
    """
    
    def __init__(self):
        """Initialize the EntityTermAnalyzer."""
        self.analyzer = LatentSpaceAnalyzer()
        self.latent_space = self.analyzer.latent_space
        
        # Create output directory if it doesn't exist
        self.output_dir = os.path.join(project_root, 'output', 'entity_term_analysis')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load entity data
        self.movements = self.analyzer.movements
        self.politicians = self.analyzer.politicians
        
        # Define important political terms for analysis
        self.political_terms = {
            'economic': [
                'wirtschaft', 'markt', 'steuern', 'haushalt', 'finanzen', 
                'wachstum', 'investition', 'handel', 'arbeit', 'industrie',
                'sozialstaat', 'umverteilung', 'wohlfahrt', 'regulierung'
            ],
            'social': [
                'gerechtigkeit', 'gleichheit', 'freiheit', 'rechte', 'familie',
                'bildung', 'gesundheit', 'sicherheit', 'integration', 'migration',
                'vielfalt', 'tradition', 'werte', 'identität', 'religion'
            ],
            'ecological': [
                'umwelt', 'klima', 'nachhaltigkeit', 'energie', 'ökologie',
                'naturschutz', 'erneuerbar', 'ressourcen', 'landwirtschaft',
                'verkehr', 'mobilität', 'atomkraft', 'kohle', 'emissionen'
            ],
            'governance': [
                'demokratie', 'staat', 'regierung', 'parlament', 'gesetz',
                'verfassung', 'föderalismus', 'europa', 'international', 'reform',
                'bürokratie', 'transparenz', 'korruption', 'partizipation'
            ]
        }
        
        # Flatten the list for easy access
        self.all_political_terms = []
        for category, terms in self.political_terms.items():
            self.all_political_terms.extend(terms)
    
    def get_entity_term_similarities(self, entity_type, entity_name):
        """
        Calculate similarities between an entity and a set of political terms.
        
        Args:
            entity_type: Type of entity ('movement' or 'politician')
            entity_name: Name of the entity
            
        Returns:
            Dictionary with term similarities by category
        """
        # Get entity embedding
        entity_embedding = self.latent_space.get_entity_embedding(entity_type, entity_name)
        if entity_embedding is None:
            print(f"Warning: No embedding found for {entity_type} {entity_name}")
            return None
        
        # Normalize the entity embedding
        entity_norm = np.linalg.norm(entity_embedding)
        if entity_norm > 0:
            # Scale to have norm around 1.7 (similar to word embeddings)
            entity_embedding = entity_embedding * (1.7 / entity_norm)
        
        # Calculate similarities for each term
        term_similarities = {}
        
        for category, terms in self.political_terms.items():
            category_similarities = {}
            
            for term in terms:
                # Get term embedding
                term_embedding = self.latent_space.word_embedding_store.get_word_embedding(term)
                
                if term_embedding is not None:
                    # Calculate cosine similarity
                    similarity = cosine_similarity(
                        entity_embedding.reshape(1, -1),
                        term_embedding.reshape(1, -1)
                    )[0][0]
                    
                    # Convert to a more intuitive scale (0-1)
                    # Cosine similarity ranges from -1 to 1, so we rescale
                    normalized_similarity = (similarity + 1) / 2
                    
                    category_similarities[term] = normalized_similarity
            
            term_similarities[category] = category_similarities
        
        return term_similarities
    
    def compare_entities_by_terms(self, entities, top_n=10):
        """
        Compare multiple entities based on their similarities to political terms.
        
        Args:
            entities: List of tuples (entity_type, entity_name)
            top_n: Number of most distinctive terms to highlight
            
        Returns:
            DataFrame with term similarities for each entity
        """
        # Get similarities for each entity
        entity_similarities = {}
        
        for entity_type, entity_name in entities:
            similarities = self.get_entity_term_similarities(entity_type, entity_name)
            if similarities:
                entity_similarities[entity_name] = similarities
        
        if not entity_similarities:
            return None
        
        # Create a DataFrame with all terms and entities
        rows = []
        
        for entity_name, categories in entity_similarities.items():
            for category, terms in categories.items():
                for term, similarity in terms.items():
                    rows.append({
                        'entity': entity_name,
                        'category': category,
                        'term': term,
                        'similarity': similarity
                    })
        
        df = pd.DataFrame(rows)
        
        # Calculate term distinctiveness
        term_distinctiveness = {}
        
        # Pivot the DataFrame to get term similarities for each entity
        pivot_df = df.pivot_table(
            index=['category', 'term'],
            columns='entity',
            values='similarity'
        )
        
        # Calculate standard deviation for each term
        pivot_df['std'] = pivot_df.std(axis=1)
        
        # Sort by standard deviation to find most distinctive terms
        pivot_df = pivot_df.sort_values('std', ascending=False)
        
        return pivot_df
    
    def get_entity_term_profile(self, entity_type, entity_name):
        """
        Create a profile of an entity based on its similarities to political terms.
        
        Args:
            entity_type: Type of entity ('movement' or 'politician')
            entity_name: Name of the entity
            
        Returns:
            Dictionary with term profile data
        """
        # Get term similarities
        term_similarities = self.get_entity_term_similarities(entity_type, entity_name)
        if not term_similarities:
            return None
        
        # Calculate category scores
        category_scores = {}
        for category, terms in term_similarities.items():
            if terms:
                category_scores[category] = sum(terms.values()) / len(terms)
            else:
                category_scores[category] = 0.0
        
        # Find top terms for each category
        top_terms = {}
        for category, terms in term_similarities.items():
            sorted_terms = sorted(terms.items(), key=lambda x: x[1], reverse=True)
            top_terms[category] = sorted_terms[:5]  # Top 5 terms
        
        return {
            'entity': {'type': entity_type, 'name': entity_name},
            'category_scores': category_scores,
            'top_terms': top_terms
        }
    
    def visualize_entity_term_profile(self, entity_type, entity_name, save_path=None):
        """
        Visualize the term profile of an entity.
        
        Args:
            entity_type: Type of entity ('movement' or 'politician')
            entity_name: Name of the entity
            save_path: Path to save the figure (if None, display instead)
            
        Returns:
            Matplotlib figure
        """
        # Get entity term profile
        profile = self.get_entity_term_profile(entity_type, entity_name)
        if not profile:
            return None
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot category scores as a radar chart
        categories = list(profile['category_scores'].keys())
        values = [profile['category_scores'][c] for c in categories]
        
        # Add the first value at the end to close the polygon
        categories.append(categories[0])
        values.append(values[0])
        
        # Calculate angles for radar chart
        angles = np.linspace(0, 2*np.pi, len(categories) - 1, endpoint=False).tolist()
        angles.append(angles[0])  # Close the circle
        
        # Plot radar chart
        ax1.plot(angles, values, 'o-', linewidth=2)
        ax1.fill(angles, values, alpha=0.25)
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels([c.capitalize() for c in categories[:-1]])
        ax1.set_title(f'Category Profile for {entity_name}')
        
        # Plot top terms as a horizontal bar chart
        terms = []
        similarities = []
        categories_list = []
        
        for category, top_terms in profile['top_terms'].items():
            for term, similarity in top_terms:
                terms.append(term)
                similarities.append(similarity)
                categories_list.append(category)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'term': terms,
            'similarity': similarities,
            'category': categories_list
        })
        
        # Sort by similarity
        df = df.sort_values('similarity', ascending=False)
        
        # Plot horizontal bar chart
        sns.barplot(
            x='similarity',
            y='term',
            hue='category',
            data=df,
            ax=ax2
        )
        ax2.set_title(f'Top Terms for {entity_name}')
        ax2.set_xlabel('Similarity')
        ax2.set_ylabel('Term')
        
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            return save_path
        else:
            return fig
    
    def visualize_entity_comparison(self, entities, save_path=None):
        """
        Visualize a comparison of multiple entities based on their term profiles.
        
        Args:
            entities: List of tuples (entity_type, entity_name)
            save_path: Path to save the figure (if None, display instead)
            
        Returns:
            Matplotlib figure
        """
        # Get profiles for each entity
        entity_profiles = {}
        for entity_type, entity_name in entities:
            profile = self.get_entity_term_profile(entity_type, entity_name)
            if profile:
                entity_profiles[entity_name] = profile
        
        if not entity_profiles:
            return None
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Plot category comparisons
        for i, category in enumerate(list(self.political_terms.keys())):
            ax = axes[i]
            
            # Extract category scores for each entity
            entities_list = []
            scores = []
            
            for entity_name, profile in entity_profiles.items():
                entities_list.append(entity_name)
                scores.append(profile['category_scores'][category])
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'entity': entities_list,
                'score': scores
            })
            
            # Sort by score
            df = df.sort_values('score', ascending=False)
            
            # Plot horizontal bar chart
            sns.barplot(
                x='score',
                y='entity',
                data=df,
                ax=ax
            )
            ax.set_title(f'{category.capitalize()} Category')
            ax.set_xlabel('Score')
            ax.set_ylabel('Entity')
        
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            return save_path
        else:
            return fig
    
    def analyze_term_associations(self, term, top_n=10):
        """
        Analyze which entities are most strongly associated with a specific term.
        
        Args:
            term: The political term to analyze
            top_n: Number of top entities to return
            
        Returns:
            DataFrame with entity-term associations
        """
        # Get term embedding
        term_embedding = self.latent_space.word_embedding_store.get_word_embedding(term)
        if term_embedding is None:
            print(f"Warning: No embedding found for term {term}")
            return None
        
        # Calculate similarities for all entities
        entity_similarities = []
        
        # Check movements
        for movement in self.movements:
            entity_embedding = self.latent_space.get_entity_embedding('movement', movement)
            if entity_embedding is not None:
                # Normalize embedding
                entity_norm = np.linalg.norm(entity_embedding)
                if entity_norm > 0:
                    entity_embedding = entity_embedding * (1.7 / entity_norm)
                
                # Calculate similarity
                similarity = cosine_similarity(
                    entity_embedding.reshape(1, -1),
                    term_embedding.reshape(1, -1)
                )[0][0]
                
                # Convert to a more intuitive scale (0-1)
                normalized_similarity = (similarity + 1) / 2
                
                entity_similarities.append({
                    'entity_type': 'movement',
                    'entity_name': movement,
                    'similarity': normalized_similarity
                })
        
        # Check politicians
        for politician in self.politicians:
            entity_embedding = self.latent_space.get_entity_embedding('politician', politician)
            if entity_embedding is not None:
                # Normalize embedding
                entity_norm = np.linalg.norm(entity_embedding)
                if entity_norm > 0:
                    entity_embedding = entity_embedding * (1.7 / entity_norm)
                
                # Calculate similarity
                similarity = cosine_similarity(
                    entity_embedding.reshape(1, -1),
                    term_embedding.reshape(1, -1)
                )[0][0]
                
                # Convert to a more intuitive scale (0-1)
                normalized_similarity = (similarity + 1) / 2
                
                entity_similarities.append({
                    'entity_type': 'politician',
                    'entity_name': politician,
                    'similarity': normalized_similarity
                })
        
        # Create DataFrame
        df = pd.DataFrame(entity_similarities)
        
        # Sort by similarity
        df = df.sort_values('similarity', ascending=False)
        
        return df.head(top_n)
    
    def visualize_term_associations(self, term, top_n=10, save_path=None):
        """
        Visualize which entities are most strongly associated with a specific term.
        
        Args:
            term: The political term to analyze
            top_n: Number of top entities to return
            save_path: Path to save the figure (if None, display instead)
            
        Returns:
            Matplotlib figure
        """
        # Get term associations
        df = self.analyze_term_associations(term, top_n)
        if df is None or len(df) == 0:
            return None
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot horizontal bar chart
        sns.barplot(
            x='similarity',
            y='entity_name',
            hue='entity_type',
            data=df
        )
        plt.title(f'Entities Most Associated with Term: {term}')
        plt.xlabel('Similarity')
        plt.ylabel('Entity')
        
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            return save_path
        else:
            return plt.gcf()
    
    def generate_term_report(self, term, output_dir=None):
        """
        Generate a comprehensive report for a specific term.
        
        Args:
            term: The political term to analyze
            output_dir: Directory to save the report files (if None, use default)
            
        Returns:
            Dictionary with paths to generated files
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, f"term_{term}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        report_files = {}
        
        # 1. Generate term associations visualization
        viz_path = os.path.join(output_dir, f"{term}_associations.png")
        self.visualize_term_associations(term, save_path=viz_path)
        report_files['associations_viz'] = viz_path
        
        # 2. Generate term associations data
        df = self.analyze_term_associations(term, top_n=20)
        if df is not None:
            data_path = os.path.join(output_dir, f"{term}_associations.csv")
            df.to_csv(data_path, index=False)
            report_files['associations_data'] = data_path
        
        # 3. Generate summary report
        summary = {
            'term': term,
            'files': report_files
        }
        
        summary_path = os.path.join(output_dir, 'summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        report_files['summary'] = summary_path
        
        return report_files
    
    def generate_entity_report(self, entity_type, entity_name, output_dir=None):
        """
        Generate a comprehensive report for a specific entity.
        
        Args:
            entity_type: Type of entity ('movement' or 'politician')
            entity_name: Name of the entity
            output_dir: Directory to save the report files (if None, use default)
            
        Returns:
            Dictionary with paths to generated files
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, f"{entity_type}_{entity_name}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        report_files = {}
        
        # 1. Generate entity term profile visualization
        profile_viz_path = os.path.join(output_dir, f"{entity_name}_profile.png")
        self.visualize_entity_term_profile(entity_type, entity_name, save_path=profile_viz_path)
        report_files['profile_viz'] = profile_viz_path
        
        # 2. Generate entity term profile data
        profile = self.get_entity_term_profile(entity_type, entity_name)
        if profile:
            profile_path = os.path.join(output_dir, f"{entity_name}_profile.json")
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
            report_files['profile_data'] = profile_path
        
        # 3. Generate word cloud
        word_cloud = self.analyzer.get_word_cloud(entity_type, entity_name, top_n=100)
        if word_cloud:
            word_cloud_path = os.path.join(output_dir, f"{entity_name}_word_cloud.json")
            with open(word_cloud_path, 'w', encoding='utf-8') as f:
                json.dump(word_cloud, f, indent=2, ensure_ascii=False)
            report_files['word_cloud'] = word_cloud_path
        
        # 4. Generate summary report
        summary = {
            'entity': {'type': entity_type, 'name': entity_name},
            'files': report_files
        }
        
        summary_path = os.path.join(output_dir, 'summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        report_files['summary'] = summary_path
        
        return report_files

def main():
    """Main function to run the entity-term analyzer."""
    parser = argparse.ArgumentParser(description='Analyze entity-term relationships in the political latent space')
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Entity profile command
    entity_parser = subparsers.add_parser('entity', help='Analyze an entity\'s term profile')
    entity_parser.add_argument('--type', choices=['movement', 'politician'], required=True,
                              help='Type of entity')
    entity_parser.add_argument('--name', required=True,
                              help='Name of entity')
    entity_parser.add_argument('--output-dir',
                              help='Directory to save output files')
    
    # Term analysis command
    term_parser = subparsers.add_parser('term', help='Analyze a term\'s entity associations')
    term_parser.add_argument('--term', required=True,
                            help='Political term to analyze')
    term_parser.add_argument('--output-dir',
                            help='Directory to save output files')
    
    # Compare entities command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple entities')
    compare_parser.add_argument('--entities', required=True, nargs='+',
                               help='Entities to compare in format type:name (e.g., movement:cdu)')
    compare_parser.add_argument('--output-dir',
                               help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = EntityTermAnalyzer()
    
    # Run command
    if args.command == 'entity':
        print(f"Analyzing entity profile for {args.type} {args.name}...")
        report = analyzer.generate_entity_report(args.type, args.name, args.output_dir)
        print(f"Report generated successfully. Files saved to: {os.path.dirname(report['summary'])}")
    
    elif args.command == 'term':
        print(f"Analyzing term associations for {args.term}...")
        report = analyzer.generate_term_report(args.term, args.output_dir)
        print(f"Report generated successfully. Files saved to: {os.path.dirname(report['summary'])}")
    
    elif args.command == 'compare':
        print(f"Comparing entities: {args.entities}...")
        
        # Parse entity strings
        entities = []
        for entity_str in args.entities:
            try:
                entity_type, entity_name = entity_str.split(':')
                entities.append((entity_type, entity_name))
            except ValueError:
                print(f"Error: Invalid entity format: {entity_str}. Use format type:name (e.g., movement:cdu)")
                return
        
        # Create output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            entity_names = '_'.join([name for _, name in entities])
            output_dir = os.path.join(analyzer.output_dir, f"compare_{entity_names}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate comparison visualization
        viz_path = os.path.join(output_dir, "entity_comparison.png")
        analyzer.visualize_entity_comparison(entities, save_path=viz_path)
        
        print(f"Comparison generated successfully. Files saved to: {output_dir}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
