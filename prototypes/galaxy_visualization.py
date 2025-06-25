"""
3D Galaxy Visualization for Political Latent Space

This script creates an interactive 3D visualization of political movements and politicians
in a galaxy-like structure, with movements as planets and politicians as moons.
"""

import json
import os
import numpy as np
import plotly.graph_objects as go
import umap
import argparse
from sklearn.preprocessing import StandardScaler
import random

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

def create_galaxy_visualization(data, output_path=None, random_seed=42, selected_entity=None):
    """
    Create an enhanced 3D galaxy visualization with dimensional axes.
    
    Args:
        data: Dictionary containing political data
        output_path: Path to save the HTML output
        random_seed: Random seed for reproducibility
        selected_entity: Dictionary with {'type': 'movement'|'politician', 'name': entity_name} for showing projections
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    # --- Data Extraction ---
    movement_embeddings = [d['embedding'] for d in data['movements'].values() if 'embedding' in d]
    movement_names = [name for name, d in data['movements'].items() if 'embedding' in d]
    politician_embeddings = [d['embedding'] for d in data.get('politicians', {}).values() if 'embedding' in d]
    politician_names = [name for name, d in data.get('politicians', {}).items() if 'embedding' in d]
    politician_movements = [d.get('movement', 'Unknown') for d in data.get('politicians', {}).values() if 'embedding' in d]

    # --- Standardize and Reduce Dimensionality ---
    # Combine all embeddings for standardization
    all_embeddings = np.vstack([movement_embeddings, politician_embeddings])
    
    # Standardize embeddings
    scaler = StandardScaler()
    all_embeddings_standardized = scaler.fit_transform(all_embeddings)
    
    # Split back into movements and politicians
    movement_embeddings_standardized = all_embeddings_standardized[:len(movement_embeddings)]
    politician_embeddings_standardized = all_embeddings_standardized[len(movement_embeddings):]

    # --- Dimensionality Reduction (UMAP) ---
    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.5, spread=2.0, random_state=random_seed)
    
    movement_positions = reducer.fit_transform(movement_embeddings_standardized)
    politician_positions = reducer.transform(politician_embeddings_standardized)

    # --- Center the data cloud around the origin (0,0,0) for camera rotation ---
    center_of_mass = np.mean(np.vstack((movement_positions, politician_positions)), axis=0)
    movement_positions -= center_of_mass
    politician_positions -= center_of_mass

    # --- Visualization Setup ---
    fig = go.Figure()
    movement_position_map = {name: pos for name, pos in zip(movement_names, movement_positions)}
    movement_colors = {name: f'hsl({(i * 60) % 360}, 80%, 50%)' for i, name in enumerate(movement_names)}

    # --- 1. Add Starry Background ---
    stars = np.random.rand(1000, 3) * 20 - 10
    fig.add_trace(go.Scatter3d(
        x=stars[:, 0], y=stars[:, 1], z=stars[:, 2],
        mode='markers', marker=dict(size=1, color='white', opacity=0.5),
        hoverinfo='none', showlegend=False
    ))

    # --- 2. Add Data-Driven Dimensional Axes ---
    all_positions = np.vstack((movement_positions, politician_positions))

    # Combine all entities with their positions and expert scores
    all_entities = []
    for name, pos in zip(movement_names, movement_positions):
        scores = data['movements'][name].get('position', {}).get('expert_dimensions', {}).get('axes', {})
        all_entities.append({'name': name, 'pos': pos, 'scores': scores, 'type': 'movement'})
    for i, pos in enumerate(politician_positions):
        name = politician_names[i]
        scores = data['politicians'][name].get('position', {}).get('expert_dimensions', {}).get('axes', {})
        all_entities.append({'name': name, 'pos': pos, 'scores': scores, 'type': 'politician'})

    # Extract dimensions from the first movement (assuming they are consistent)
    first_movement = next(iter(data['movements'].values()))
    dimensions = list(first_movement.get('position', {}).get('expert_dimensions', {}).get('axes', {}).keys())
    axes_to_draw = dimensions[:3] # Limit to 3 axes for clarity

    max_range = np.linalg.norm(all_positions, axis=1).max()

    # Define colors for each dimension axis
    axis_colors = {
        'economic_axis': 'rgba(255, 165, 0, 0.8)',  # Orange
        'social_axis': 'rgba(65, 105, 225, 0.8)',    # Royal Blue
        'ecological_axis': 'rgba(50, 205, 50, 0.8)'  # Lime Green
    }
    
    # Store axis vectors for projection calculations
    axis_vectors = {}
    
    for dim_name in axes_to_draw:
        # Find the average position of the top 20% of entities for this dimension
        all_entities.sort(key=lambda x: x['scores'].get(dim_name, -1), reverse=True)
        top_entities = all_entities[:int(len(all_entities) * 0.2)]
        if not top_entities:
            continue
        
        direction_point = np.mean([e['pos'] for e in top_entities], axis=0)
        vector = direction_point
        unit_vector = vector / np.linalg.norm(vector)  # Normalized vector
        scaled_vector = unit_vector * max_range  # Scaled for visualization
        
        # Store the unit vector for projection calculations
        axis_vectors[dim_name] = unit_vector

        axis_end = scaled_vector
        axis_color = axis_colors.get(dim_name, 'rgba(255, 255, 255, 0.8)')

        # Draw axis line in both positive and negative directions from center of mass
        negative_end = -scaled_vector
        
        fig.add_trace(go.Scatter3d(
            x=[negative_end[0], axis_end[0]], 
            y=[negative_end[1], axis_end[1]], 
            z=[negative_end[2], axis_end[2]],
            mode='lines', line=dict(color=axis_color, width=3),
            hoverinfo='none', showlegend=False
        ))
        
        # Add axis labels at both ends of the line
        fig.add_trace(go.Scatter3d(
            x=[axis_end[0] * 1.1], y=[axis_end[1] * 1.1], z=[axis_end[2] * 1.1],
            mode='text', text=[dim_name.replace('_', ' ').title() + ' +'],
            textfont=dict(color=axis_color, size=14),
            hoverinfo='none', showlegend=False
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[negative_end[0] * 1.1], y=[negative_end[1] * 1.1], z=[negative_end[2] * 1.1],
            mode='text', text=[dim_name.replace('_', ' ').title() + ' -'],
            textfont=dict(color=axis_color, size=14),
            hoverinfo='none', showlegend=False
        ))
        
        # Add tick marks along the axis in both positive and negative directions
        for tick_val in np.arange(0.2, 1.2, 0.2):
            # Positive direction tick
            pos_tick_pos = unit_vector * max_range * tick_val
            
            # Add positive tick mark
            fig.add_trace(go.Scatter3d(
                x=[pos_tick_pos[0]], y=[pos_tick_pos[1]], z=[pos_tick_pos[2]],
                mode='markers',
                marker=dict(size=5, color=axis_color),
                hoverinfo='none', showlegend=False
            ))
            
            # Add positive tick label with offset perpendicular to axis
            # Calculate offset direction perpendicular to the axis
            if abs(unit_vector[0]) < 0.9:  # If not aligned with x-axis
                perp_vector = np.cross(unit_vector, [1, 0, 0])
            else:
                perp_vector = np.cross(unit_vector, [0, 1, 0])
            perp_vector = perp_vector / np.linalg.norm(perp_vector) * 0.05 * max_range
            
            fig.add_trace(go.Scatter3d(
                x=[pos_tick_pos[0] + perp_vector[0]], 
                y=[pos_tick_pos[1] + perp_vector[1]], 
                z=[pos_tick_pos[2] + perp_vector[2]],
                mode='text',
                text=[f'{tick_val:.1f}'],
                textfont=dict(color=axis_color, size=10),
                hoverinfo='none', showlegend=False
            ))
            
            # Negative direction tick
            neg_tick_pos = -unit_vector * max_range * tick_val
            
            # Add negative tick mark
            fig.add_trace(go.Scatter3d(
                x=[neg_tick_pos[0]], y=[neg_tick_pos[1]], z=[neg_tick_pos[2]],
                mode='markers',
                marker=dict(size=5, color=axis_color),
                hoverinfo='none', showlegend=False
            ))
            
            # Add negative tick label
            fig.add_trace(go.Scatter3d(
                x=[neg_tick_pos[0] + perp_vector[0]], 
                y=[neg_tick_pos[1] + perp_vector[1]], 
                z=[neg_tick_pos[2] + perp_vector[2]],
                mode='text',
                text=[f'{tick_val:.1f}'],
                textfont=dict(color=axis_color, size=10),
                hoverinfo='none', showlegend=False
            ))

    # --- 3. Add Movements (Planets) ---
    # Create enhanced hover templates with dimension scores
    movement_hovertemplates = []
    for name in movement_names:
        movement_data = data['movements'][name]
        dim_scores = movement_data.get('position', {}).get('expert_dimensions', {}).get('axes', {})
        
        hovertext = f"<b>{name}</b><br>(Movement)<br><br>"
        hovertext += "<b>Dimension Scores:</b><br>"
        for dim in dimensions[:5]:  # Show top 5 dimensions
            score = dim_scores.get(dim, 0)
            hovertext += f"{dim.replace('_', ' ').title()}: {score:.2f}<br>"
        
        movement_hovertemplates.append(hovertext + "<extra></extra>")
    
    fig.add_trace(go.Scatter3d(
        x=movement_positions[:, 0], y=movement_positions[:, 1], z=movement_positions[:, 2],
        mode='markers+text', text=movement_names, textposition='top center',
        marker=dict(size=12, color=[movement_colors.get(m, 'grey') for m in movement_names], opacity=0.9, line=dict(color='white', width=1)),
        name='Movements', hovertemplate=movement_hovertemplates, hoverinfo='text'
    ))

    # --- 4. Add Politicians (Moons) ---
    # Group politicians by movement for more efficient plotting
    politician_traces = {}
    
    for i, pos in enumerate(politician_positions):
        movement = politician_movements[i]
        name = politician_names[i]
        final_pos = pos # Simplified positioning for clarity
        
        # Create enhanced hover template with dimension scores
        politician_data = data['politicians'][name]
        dim_scores = politician_data.get('position', {}).get('expert_dimensions', {}).get('axes', {})
        
        hovertext = f"<b>{name}</b><br>{movement}<br><br>"
        hovertext += "<b>Dimension Scores:</b><br>"
        for dim in dimensions[:5]:  # Show top 5 dimensions
            score = dim_scores.get(dim, 0)
            hovertext += f"{dim.replace('_', ' ').title()}: {score:.2f}<br>"
        
        hovertemplate = hovertext + "<extra></extra>"
        
        # Group by movement for more efficient plotting
        if movement not in politician_traces:
            politician_traces[movement] = {
                'x': [], 'y': [], 'z': [], 
                'names': [], 'hovertemplates': []
            }
            
        politician_traces[movement]['x'].append(final_pos[0])
        politician_traces[movement]['y'].append(final_pos[1])
        politician_traces[movement]['z'].append(final_pos[2])
        politician_traces[movement]['names'].append(name)
        politician_traces[movement]['hovertemplates'].append(hovertemplate)
    
    # Add traces for each movement's politicians
    for movement, trace_data in politician_traces.items():
        fig.add_trace(go.Scatter3d(
            x=trace_data['x'], y=trace_data['y'], z=trace_data['z'],
            mode='markers', marker=dict(size=6, color=movement_colors.get(movement, 'grey'), opacity=0.8),
            name=movement, text=trace_data['names'],
            hovertemplate=trace_data['hovertemplates'], hoverinfo='text',
            showlegend=False
        ))

    # --- 5. Add Projection Lines for Selected Entity ---
    if selected_entity:
        # Find the selected entity
        selected_pos = None
        selected_name = selected_entity.get('name')
        entity_type = selected_entity.get('type')
        
        if entity_type == 'movement' and selected_name in movement_position_map:
            selected_pos = movement_position_map[selected_name]
        elif entity_type == 'politician':
            for i, name in enumerate(politician_names):
                if name == selected_name:
                    selected_pos = politician_positions[i]
                    break
        
        if selected_pos is not None:
            # Draw projection lines to each axis
            for dim_name in axes_to_draw:
                if dim_name not in axis_vectors:
                    continue
                    
                # Get the axis unit vector
                axis_vector = axis_vectors[dim_name]
                
                # Calculate projection point
                # Formula: projection = center_of_mass + (dot(point - center, axis_vector) * axis_vector)
                point_centered = selected_pos - center_of_mass
                projection_scalar = np.dot(point_centered, axis_vector)
                projection_point = center_of_mass + projection_scalar * axis_vector
                
                # Get axis color
                axis_color = axis_colors.get(dim_name, 'rgba(255, 255, 255, 0.8)')
                
                # Draw projection line
                fig.add_trace(go.Scatter3d(
                    x=[selected_pos[0], projection_point[0]], 
                    y=[selected_pos[1], projection_point[1]], 
                    z=[selected_pos[2], projection_point[2]],
                    mode='lines', 
                    line=dict(color=axis_color, width=2, dash='dash'),
                    hoverinfo='none', 
                    showlegend=False
                ))
                
                # Add marker at projection point
                fig.add_trace(go.Scatter3d(
                    x=[projection_point[0]], 
                    y=[projection_point[1]], 
                    z=[projection_point[2]],
                    mode='markers+text', 
                    marker=dict(size=5, color=axis_color),
                    text=[f"{projection_scalar:.2f}"],
                    textposition="middle right",
                    textfont=dict(color=axis_color),
                    hoverinfo='text', 
                    hovertext=f"{dim_name.replace('_', ' ').title()}: {projection_scalar:.2f}",
                    showlegend=False
                ))
    
    # --- Layout and Final Touches ---
    # Camera distance from center
    camera_distance = 1.8
    
    fig.update_layout(
        title_text='Political Latent Space', template='plotly_dark',
        scene=dict(
            xaxis_title='', yaxis_title='', zaxis_title='',
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False),
            bgcolor='rgb(10, 10, 30)',
            # Since data is centered at origin, camera naturally rotates around center of mass
            camera=dict(
                eye=dict(x=camera_distance, y=camera_distance, z=camera_distance),
                center=dict(x=0, y=0, z=0)  # Look at origin where center of mass is now located
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    if output_path:
        fig.write_html(output_path)
    
    return fig

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Create 3D galaxy visualization')
    parser.add_argument('--input', type=str, default='src/data/processed/political_latent_space.json',
                        help='Path to input JSON file')
    parser.add_argument('--output', type=str, default='prototypes/visualizations/political_galaxy_3d.html',
                        help='Path to output HTML file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--select', type=str, default=None,
                        help='Name of entity to show projections for (format: type:name, e.g. movement:cdu)')
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input}...")
    data = load_data(args.input)
    
    # Create visualization
    print("Creating 3D galaxy visualization...")
    create_galaxy_visualization(data, args.output, args.seed)
    
    print("Galaxy visualization complete.")

if __name__ == '__main__':
    main()
