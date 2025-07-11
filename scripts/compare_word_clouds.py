#!/usr/bin/env python3
"""
Word Cloud Comparison Tool

This script provides utilities to compare word clouds between political entities
and visualize the results.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import argparse
from matplotlib_venn import venn2, venn3
from wordcloud import WordCloud
import matplotlib.colors as mcolors

# Add the project root to the path so we can import the modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our analyzer
from scripts.analyze_latent_space import LatentSpaceAnalyzer

class WordCloudVisualizer:
    """
    A class for visualizing word cloud comparisons between political entities.
    """
    
    def __init__(self):
        """Initialize the WordCloudVisualizer."""
        self.analyzer = LatentSpaceAnalyzer()
        
        # Set up a colorful palette for visualizations
        self.colors = list(mcolors.TABLEAU_COLORS.values())
        
        # Create output directory if it doesn't exist
        self.output_dir = os.path.join(project_root, 'output', 'word_cloud_analysis')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def visualize_venn_diagram(self, entities, top_n=50, max_distance=0.5, save_path=None):
        """
        Create a Venn diagram showing word overlap between entities.
        
        Args:
            entities: List of tuples (entity_type, entity_name)
            top_n: Number of words to include for each entity
            max_distance: Maximum distance threshold
            save_path: Path to save the figure (if None, display instead)
            
        Returns:
            Matplotlib figure
        """
        if len(entities) < 2 or len(entities) > 3:
            raise ValueError("Venn diagram requires 2 or 3 entities")
        
        # Get word clouds for each entity
        entity_words = []
        entity_labels = []
        
        for entity_type, entity_name in entities:
            cloud = self.analyzer.get_word_cloud(entity_type, entity_name, top_n=top_n, max_distance=max_distance)
            words = {item['text'] for item in cloud}
            entity_words.append(words)
            entity_labels.append(f"{entity_name}")
        
        # Create figure
        plt.figure(figsize=(10, 7))
        
        # Create Venn diagram
        if len(entities) == 2:
            venn = venn2(entity_words, entity_labels)
            # Set colors
            for i, patch in enumerate(venn.patches):
                if patch:
                    patch.set_color(self.colors[i % len(self.colors)])
        else:  # 3 entities
            venn = venn3(entity_words, entity_labels)
            # Set colors
            for i, patch in enumerate(venn.patches):
                if patch:
                    patch.set_color(self.colors[i % len(self.colors)])
        
        # Set title
        entity_names = [name for _, name in entities]
        plt.title(f"Word Cloud Overlap: {', '.join(entity_names)}")
        
        # Save or display
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            return save_path
        else:
            plt.tight_layout()
            return plt.gcf()
    
    def visualize_word_clouds(self, entities, top_n=50, max_distance=0.5, save_path=None):
        """
        Create word clouds for each entity and a combined differential cloud.
        
        Args:
            entities: List of tuples (entity_type, entity_name)
            top_n: Number of words to include for each entity
            max_distance: Maximum distance threshold
            save_path: Path to save the figure (if None, display instead)
            
        Returns:
            Matplotlib figure
        """
        if len(entities) < 1 or len(entities) > 3:
            raise ValueError("Can visualize 1-3 entities at a time")
        
        # Get word clouds for each entity
        entity_clouds = []
        entity_names = []
        
        for entity_type, entity_name in entities:
            cloud = self.analyzer.get_word_cloud(entity_type, entity_name, top_n=top_n, max_distance=max_distance)
            entity_clouds.append({item['text']: item['value'] for item in cloud})
            entity_names.append(entity_name)
        
        # Create figure
        n_cols = min(len(entities), 3)
        fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
        
        # Handle single entity case
        if n_cols == 1:
            axes = [axes]
        
        # Generate word clouds
        for i, (name, word_dict) in enumerate(zip(entity_names, entity_clouds)):
            # Create WordCloud object
            wc = WordCloud(
                width=800, 
                height=400,
                background_color='white',
                colormap='viridis',
                max_words=top_n,
                prefer_horizontal=0.9
            ).generate_from_frequencies(word_dict)
            
            # Display
            axes[i].imshow(wc, interpolation='bilinear')
            axes[i].set_title(name)
            axes[i].axis('off')
        
        # Set overall title
        plt.suptitle(f"Word Clouds: {', '.join(entity_names)}", fontsize=16)
        
        # Save or display
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            return save_path
        else:
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
            return plt.gcf()
    
    def visualize_distinctive_words(self, entity, compare_to=None, top_n=20, max_distance=0.5, save_path=None):
        """
        Visualize the most distinctive words for an entity.
        
        Args:
            entity: Tuple of (entity_type, entity_name)
            compare_to: List of entities to compare against (if None, compare to all of same type)
            top_n: Number of words to include
            max_distance: Maximum distance threshold
            save_path: Path to save the figure (if None, display instead)
            
        Returns:
            Matplotlib figure
        """
        entity_type, entity_name = entity
        
        # Get distinctive words
        distinctive_words = self.analyzer.find_distinctive_words(
            entity_type, entity_name, compare_to=compare_to, 
            top_n=top_n, max_distance=max_distance
        )
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Extract data for plotting
        words = [item['text'] for item in distinctive_words]
        distinctiveness = [item['distinctiveness'] for item in distinctive_words]
        
        # Create horizontal bar chart
        bars = plt.barh(words[::-1], distinctiveness[::-1], color=self.colors[0])
        
        # Add labels
        plt.xlabel('Distinctiveness Score')
        plt.title(f'Most Distinctive Words for {entity_name}')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(
                width + 0.01,
                bar.get_y() + bar.get_height()/2,
                f'{width:.2f}',
                va='center'
            )
        
        # Save or display
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            return save_path
        else:
            plt.tight_layout()
            return plt.gcf()
    
    def visualize_political_dimensions(self, entities, top_n=50, save_path=None):
        """
        Visualize the political dimensions of entities based on their word clouds.
        
        Args:
            entities: List of tuples (entity_type, entity_name)
            top_n: Number of words to include
            save_path: Path to save the figure (if None, display instead)
            
        Returns:
            Matplotlib figure
        """
        # Get political dimensions for each entity
        dimension_data = []
        entity_names = []
        
        for entity_type, entity_name in entities:
            dimensions = self.analyzer.analyze_political_dimensions(entity_type, entity_name, top_n=top_n)
            dimension_data.append(dimensions['composite_scores'])
            entity_names.append(entity_name)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set up dimensions for radar chart
        dimensions = list(dimension_data[0].keys())
        n_dims = len(dimensions)
        angles = np.linspace(0, 2*np.pi, n_dims, endpoint=False).tolist()
        angles += angles[:1]  # Close the circle
        
        # Set up radar chart
        ax = plt.subplot(111, polar=True)
        
        # Plot each entity
        for i, (name, data) in enumerate(zip(entity_names, dimension_data)):
            values = [data[dim] for dim in dimensions]
            values += values[:1]  # Close the circle
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, label=name, color=self.colors[i % len(self.colors)])
            ax.fill(angles, values, alpha=0.1, color=self.colors[i % len(self.colors)])
        
        # Set up radar chart labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([dim.replace('_', ' ').title() for dim in dimensions])
        
        # Add legend and title
        plt.legend(loc='upper right')
        plt.title('Political Dimensions Comparison')
        
        # Save or display
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            return save_path
        else:
            plt.tight_layout()
            return plt.gcf()
    
    def generate_comparison_report(self, entities, output_dir=None, top_n=50, max_distance=0.5):
        """
        Generate a comprehensive comparison report for the given entities.
        
        Args:
            entities: List of tuples (entity_type, entity_name)
            output_dir: Directory to save the report files (if None, use default)
            top_n: Number of words to include
            max_distance: Maximum distance threshold
            
        Returns:
            Dictionary with paths to generated files
        """
        if output_dir is None:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            entity_names = '_'.join([name for _, name in entities])
            output_dir = os.path.join(self.output_dir, f"{timestamp}_{entity_names}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        report_files = {}
        
        # 1. Generate word clouds
        wc_path = os.path.join(output_dir, 'word_clouds.png')
        self.visualize_word_clouds(entities, top_n=top_n, max_distance=max_distance, save_path=wc_path)
        report_files['word_clouds'] = wc_path
        
        # 2. Generate Venn diagram if 2-3 entities
        if 2 <= len(entities) <= 3:
            venn_path = os.path.join(output_dir, 'venn_diagram.png')
            self.visualize_venn_diagram(entities, top_n=top_n, max_distance=max_distance, save_path=venn_path)
            report_files['venn_diagram'] = venn_path
        
        # 3. Generate distinctive words for each entity
        for entity_type, entity_name in entities:
            dist_path = os.path.join(output_dir, f'distinctive_{entity_name}.png')
            self.visualize_distinctive_words((entity_type, entity_name), top_n=top_n, save_path=dist_path)
            report_files[f'distinctive_{entity_name}'] = dist_path
        
        # 4. Generate political dimensions comparison
        dim_path = os.path.join(output_dir, 'political_dimensions.png')
        self.visualize_political_dimensions(entities, top_n=top_n, save_path=dim_path)
        report_files['political_dimensions'] = dim_path
        
        # 5. Generate pairwise comparisons if more than 1 entity
        if len(entities) > 1:
            for i, (type1, name1) in enumerate(entities):
                for j, (type2, name2) in enumerate(entities[i+1:], i+1):
                    comparison = self.analyzer.compare_word_clouds(
                        (type1, name1), (type2, name2), 
                        top_n=top_n, max_distance=max_distance
                    )
                    
                    # Save comparison as JSON
                    comp_path = os.path.join(output_dir, f'comparison_{name1}_{name2}.json')
                    with open(comp_path, 'w', encoding='utf-8') as f:
                        json.dump(comparison, f, indent=2, ensure_ascii=False)
                    
                    report_files[f'comparison_{name1}_{name2}'] = comp_path
        
        # 6. Generate summary report
        summary = {
            'entities': [{'type': t, 'name': n} for t, n in entities],
            'parameters': {
                'top_n': top_n,
                'max_distance': max_distance
            },
            'files': report_files
        }
        
        summary_path = os.path.join(output_dir, 'summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        report_files['summary'] = summary_path
        
        return report_files

def main():
    """Main function to run the word cloud comparison tool."""
    parser = argparse.ArgumentParser(description='Compare word clouds between political entities')
    
    # Entity arguments
    parser.add_argument('--entity1-type', choices=['movement', 'politician'], required=True,
                        help='Type of the first entity')
    parser.add_argument('--entity1-name', required=True,
                        help='Name of the first entity')
    parser.add_argument('--entity2-type', choices=['movement', 'politician'],
                        help='Type of the second entity')
    parser.add_argument('--entity2-name',
                        help='Name of the second entity')
    parser.add_argument('--entity3-type', choices=['movement', 'politician'],
                        help='Type of the third entity')
    parser.add_argument('--entity3-name',
                        help='Name of the third entity')
    
    # Analysis parameters
    parser.add_argument('--top-n', type=int, default=50,
                        help='Number of words to include')
    parser.add_argument('--max-distance', type=float, default=0.5,
                        help='Maximum distance threshold')
    parser.add_argument('--output-dir',
                        help='Directory to save the report files')
    
    args = parser.parse_args()
    
    # Collect entities
    entities = [(args.entity1_type, args.entity1_name)]
    
    if args.entity2_name and args.entity2_type:
        entities.append((args.entity2_type, args.entity2_name))
    
    if args.entity3_name and args.entity3_type:
        entities.append((args.entity3_type, args.entity3_name))
    
    # Run analysis
    visualizer = WordCloudVisualizer()
    report_files = visualizer.generate_comparison_report(
        entities,
        output_dir=args.output_dir,
        top_n=args.top_n,
        max_distance=args.max_distance
    )
    
    print(f"Report generated successfully. Files saved to: {os.path.dirname(report_files['summary'])}")
    print(f"Summary file: {report_files['summary']}")

if __name__ == "__main__":
    main()
