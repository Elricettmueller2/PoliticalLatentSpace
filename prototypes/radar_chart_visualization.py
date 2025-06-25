"""
Radar Chart Visualization for Political Latent Space

This script creates interactive radar charts comparing political movements and politicians
across expert-defined dimensions from the processed JSON data.
"""

import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import argparse

def load_data(json_path):
    """
    Load the processed political latent space data from JSON.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        Dictionary containing the processed data
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_radar_chart(entities, entity_data, dimensions, title, output_path=None):
    """
    Create a radar chart comparing multiple entities across dimensions.
    
    Args:
        entities: List of entity names to include in the chart
        entity_data: Dictionary containing entity data with expert_dimensions
        dimensions: List of dimension names to include in the chart
        title: Title for the chart
        output_path: Optional path to save the HTML output
        
    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add traces for each entity
    for i, entity in enumerate(entities):
        if entity not in entity_data:
            print(f"Warning: {entity} not found in data. Skipping.")
            continue
            
        # Get dimension values for this entity
        if 'position' not in entity_data[entity] or 'expert_dimensions' not in entity_data[entity]['position'] or 'axes' not in entity_data[entity]['position']['expert_dimensions']:
            print(f"Warning: No expert dimensions found for {entity}. Skipping.")
            continue
            
        values = []
        for dim in dimensions:
            if dim in entity_data[entity]['position']['expert_dimensions']['axes']:
                # Scale from -1,1 to 0,1 for radar chart
                values.append((entity_data[entity]['position']['expert_dimensions']['axes'][dim] + 1) / 2)
            else:
                values.append(0)
                
        # Close the polygon by repeating the first value
        values.append(values[0])
        dimensions_closed = dimensions + [dimensions[0]]
        
        # Generate a color based on the entity index
        color = f'hsl({(i * 30) % 360}, 70%, 50%)'
        
        # Add trace
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=dimensions_closed,
            fill='toself',
            name=entity,
            line=dict(color=color),
            fillcolor=color,
            opacity=0.6
        ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20)
        ),
        legend=dict(
            yanchor="top",
            y=1.1,
            xanchor="left",
            x=0.01
        ),
        template="plotly_dark"
    )
    
    # Save if output path is provided
    if output_path:
        fig.write_html(output_path)
        print(f"Radar chart saved to {output_path}")
    
    return fig

def create_comparison_charts(data, output_dir="visualizations"):
    """
    Create multiple comparison charts:
    1. All movements comparison
    2. Selected politicians comparison
    3. Movement vs. its politicians comparisons
    
    Args:
        data: Loaded JSON data
        output_dir: Directory to save output HTML files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all available dimensions
    dimensions = []
    for movement in data['movements'].values():
        if 'position' in movement and 'expert_dimensions' in movement['position'] and 'axes' in movement['position']['expert_dimensions']:
            dimensions = list(movement['position']['expert_dimensions']['axes'].keys())
            break
    
    if not dimensions:
        print("Error: No expert dimensions found in the data.")
        return
    
    # 1. Create chart comparing all movements
    movement_names = list(data['movements'].keys())
    create_radar_chart(
        movement_names, 
        data['movements'], 
        dimensions, 
        "Political Movements Comparison",
        os.path.join(output_dir, "movements_comparison.html")
    )
    
    # 2. Create chart comparing selected politicians
    # Select politicians from different movements for comparison
    selected_politicians = []
    movements_covered = set()
    
    for politician, pol_data in data['politicians'].items():
        movement = pol_data.get('movement')
        if movement and movement not in movements_covered:
            selected_politicians.append(politician)
            movements_covered.add(movement)
            
            # Limit to 5 politicians for readability
            if len(selected_politicians) >= 5:
                break
    
    create_radar_chart(
        selected_politicians, 
        data['politicians'], 
        dimensions, 
        "Selected Politicians Comparison",
        os.path.join(output_dir, "politicians_comparison.html")
    )
    
    # 3. Create charts comparing each movement with its politicians
    for movement in movement_names:
        # Find politicians belonging to this movement
        movement_politicians = [p for p, p_data in data['politicians'].items() 
                               if p_data.get('movement') == movement]
        
        if movement_politicians:
            # Combine movement with its politicians
            entities_to_compare = [movement] + movement_politicians
            
            # Create combined data dictionary
            combined_data = {}
            combined_data.update(data['movements'])
            combined_data.update(data['politicians'])
            
            create_radar_chart(
                entities_to_compare,
                combined_data,
                dimensions,
                f"{movement} and its Politicians",
                os.path.join(output_dir, f"{movement}_politicians.html")
            )

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Create radar chart visualizations')
    parser.add_argument('--input', type=str, default='src/data/processed/political_latent_space.json',
                        help='Path to input JSON file')
    parser.add_argument('--output-dir', type=str, default='prototypes/visualizations',
                        help='Directory to save output HTML files')
    args = parser.parse_args()
    
    # Load data
    data = load_data(args.input)
    
    # Create visualizations
    create_comparison_charts(data, args.output_dir)
    
    print("Radar chart visualizations complete.")

if __name__ == '__main__':
    main()
