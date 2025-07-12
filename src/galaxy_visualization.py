"""
3D Galaxy Visualization for Political Latent Space

This script creates an interactive 3D visualization of political movements and politicians
in a galaxy-like structure, with movements as planets and politicians as moons.
"""

import json
import os
import numpy as np
import plotly.graph_objects as go
from umap import UMAP
import argparse
from sklearn.preprocessing import StandardScaler
import random
import sys

# Add parent directory to path to import from app.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from app import WORD_DIMENSION_SCORES
except ImportError:
    # Fallback if import fails
    WORD_DIMENSION_SCORES = {}

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

def create_galaxy_visualization(data, output_path=None, random_seed=42, selected_entity=None, hybrid_mode=False, num_words=50):
    """
    Create a 3D visualization of the political latent space with fixed orthogonal axes
    representing the German political landscape.
    
    Parameters:
    - data: Dictionary containing movement and politician data
    - output_path: Path to save the HTML output (optional)
    - random_seed: Random seed for reproducibility
    - selected_entity: Dictionary with 'type' and 'name' of the entity to highlight with projection lines
    - hybrid_mode: If True, creates a focused visualization on the selected entity and its relationships
    - num_words: Number of words to include in the word cloud (for hybrid mode)
    
    Returns:
    - Plotly figure object
    """
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
    # Print the keys in the data structure for debugging
    print(f"Data keys: {list(data.keys() if isinstance(data, dict) else [])}")
    
    # Create empty lists as defaults
    movement_embeddings = []
    movement_names = []
    politician_embeddings = []
    politician_names = []
    politician_movements = []
    
    # Extract movement data if available
    if isinstance(data, dict) and 'movements' in data and data['movements']:
        print(f"Found {len(data['movements'])} movements")
        for name, d in data['movements'].items():
            if isinstance(d, dict) and 'embedding' in d and d['embedding']:
                movement_embeddings.append(d['embedding'])
                movement_names.append(name)
    
    # Extract politician data if available
    if isinstance(data, dict) and 'politicians' in data and data['politicians']:
        print(f"Found {len(data['politicians'])} politicians")
        for name, d in data['politicians'].items():
            if isinstance(d, dict) and 'embedding' in d and d['embedding']:
                politician_embeddings.append(d['embedding'])
                politician_names.append(name)
                politician_movements.append(d.get('movement', 'Unknown'))

    # --- Standardize and Reduce Dimensionality ---
    # Check if we have any embeddings to work with
    if len(movement_embeddings) == 0 and len(politician_embeddings) == 0:
        # No embeddings available, create a simple placeholder visualization
        print("No embeddings available, creating placeholder visualization")
        fig = go.Figure()
        
        # Add a text annotation explaining the issue
        fig.update_layout(
            title="No Embeddings Available",
            annotations=[{
                'text': "No embeddings data available. Please check your data source.",
                'showarrow': False,
                'font': {'size': 16},
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5
            }],
            template='plotly_dark',
            paper_bgcolor='rgb(10,10,25)',
            plot_bgcolor='rgb(10,10,25)'
        )
        return fig
    
    # Combine all embeddings for standardization
    if len(movement_embeddings) > 0 or len(politician_embeddings) > 0:
        # Check if we have any embeddings with valid dimensions
        if len(movement_embeddings) > 0:
            embedding_dim = len(movement_embeddings[0])
        else:
            embedding_dim = len(politician_embeddings[0])
            
        # Stack the embeddings
        all_embeddings = np.vstack([movement_embeddings, politician_embeddings])
        
        # Standardize embeddings
        scaler = StandardScaler()
        all_embeddings_standardized = scaler.fit_transform(all_embeddings)
    else:
        # Create placeholder data if no embeddings are available
        print("No embeddings available, creating placeholder data")
        embedding_dim = 3  # Default dimension
        all_embeddings = np.zeros((1, embedding_dim))
        all_embeddings_standardized = all_embeddings.copy()
    
    # Split back into movements and politicians
    movement_embeddings_standardized = all_embeddings_standardized[:len(movement_embeddings)]
    politician_embeddings_standardized = all_embeddings_standardized[len(movement_embeddings):]

    # --- Direct Mapping to Political Dimensions ---
    # Instead of using UMAP, we'll directly map entities to our defined political dimensions
    # based on their expert dimension scores
    print("Using expert dimension scores for direct political dimension mapping")
    
    # Define the political dimensions we want to use
    political_dimensions = ['economic_axis', 'social_axis', 'ecological_axis']
    
    # Create empty arrays for positions
    movement_positions = np.zeros((len(movement_names), 3))
    politician_positions = np.zeros((len(politician_names), 3))
    
    # Map movements to positions based on expert dimension scores
    for i, name in enumerate(movement_names):
        movement_data = data['movements'][name]
        dim_scores = movement_data.get('position', {}).get('expert_dimensions', {}).get('axes', {})
        
        # Default position at origin if no scores available
        position = [0, 0, 0]
        
        # Map each dimension to its corresponding axis
        for dim_idx, dim_name in enumerate(political_dimensions):
            if dim_idx < 3 and dim_name in dim_scores:  # Ensure we only use first 3 dimensions
                # Convert score from 0-1 range to -1 to 1 range for better visualization
                # (assuming original scores are in 0-1 range)
                score = dim_scores[dim_name]
                position[dim_idx] = score * 2 - 1  # Convert 0-1 to -1-1
        
        movement_positions[i] = position
    
    # Map politicians to positions based on expert dimension scores
    for i, name in enumerate(politician_names):
        politician_data = data['politicians'][name]
        dim_scores = politician_data.get('position', {}).get('expert_dimensions', {}).get('axes', {})
        
        # Default position at origin if no scores available
        position = [0, 0, 0]
        
        # Map each dimension to its corresponding axis
        for dim_idx, dim_name in enumerate(political_dimensions):
            if dim_idx < 3 and dim_name in dim_scores:  # Ensure we only use first 3 dimensions
                # Convert score from 0-1 range to -1 to 1 range for better visualization
                score = dim_scores[dim_name]
                position[dim_idx] = score * 2 - 1  # Convert 0-1 to -1-1
        
        politician_positions[i] = position
    
    # If we have no valid positions (all zeros), add some random jitter to avoid everything at origin
    if np.all(movement_positions == 0) and np.all(politician_positions == 0):
        print("No expert dimension scores found, adding random jitter")
        movement_positions = np.random.uniform(-0.5, 0.5, movement_positions.shape)
        politician_positions = np.random.uniform(-0.5, 0.5, politician_positions.shape)
    
    # Scale the positions to a reasonable range for visualization
    max_range = 1.0  # Keep positions in a standardized -1 to 1 range

    # --- Visualization Setup ---
    fig = go.Figure()
    movement_position_map = {name: pos for name, pos in zip(movement_names, movement_positions)}
    
    # Define official party colors for German political parties
    party_colors = {
        # Original case versions
        'AfD': 'rgb(0, 102, 153)',
        'CDU': 'rgb(0, 0, 0)',
        'SPD': 'rgb(230, 0, 27)',
        'LINKE': 'rgb(112, 0, 58)',
        'CSU': 'rgb(0, 126, 199)',
        'FDP': 'rgb(255, 240, 23)',
        'Grüne': 'rgb(5, 150, 39)',
        'Die Grünen': 'rgb(5, 150, 39)',
        'Bündnis 90/Die Grünen': 'rgb(5, 150, 39)',
        
        # Lowercase versions (as seen in the visualization)
        'afd': 'rgb(0, 102, 153)',
        'cdu': 'rgb(0, 0, 0)',
        'spd': 'rgb(230, 0, 27)',
        'linke': 'rgb(112, 0, 58)',
        'csu': 'rgb(0, 126, 199)',
        'fdp': 'rgb(255, 240, 23)',
        'grüne': 'rgb(5, 150, 39)'
    }
    
    # Assign colors to movements, using official party colors where available
    # and falling back to a default color scheme for other movements
    movement_colors = {}
    for name in movement_names:
        # Check for exact match or case-insensitive partial match
        exact_match = party_colors.get(name)
        if exact_match:
            movement_colors[name] = exact_match
            continue
            
        # Try to find a partial match (e.g., if 'CDU/CSU' should match 'CDU')
        matched = False
        for party, color in party_colors.items():
            if party in name or name in party:
                movement_colors[name] = color
                matched = True
                break
                
        # Fall back to default color scheme if no match found
        if not matched:
            # Generate a color based on the index to ensure consistency
            idx = movement_names.index(name)
            movement_colors[name] = f'hsl({(idx * 60) % 360}, 80%, 50%)'
    
    # --- Handle Hybrid Mode ---
    # If hybrid mode is enabled and we have a selected entity, focus the visualization on that entity
    focused_entity = None
    focused_entity_position = None
    related_entities = []
    
    if hybrid_mode and selected_entity:
        entity_type = selected_entity.get('type')
        entity_name = selected_entity.get('name')
        
        # Find the selected entity and its position
        if entity_type == 'movement' and entity_name in movement_names:
            idx = movement_names.index(entity_name)
            focused_entity = {'name': entity_name, 'type': 'movement', 'position': movement_positions[idx]}
            focused_entity_position = movement_positions[idx]
            
            # Find related politicians (those belonging to this movement)
            for i, politician_movement in enumerate(politician_movements):
                if politician_movement == entity_name:
                    related_entities.append({
                        'name': politician_names[i],
                        'type': 'politician',
                        'position': politician_positions[i],
                        'relation': 'member'
                    })
                    
        elif entity_type == 'politician' and entity_name in politician_names:
            idx = politician_names.index(entity_name)
            focused_entity = {'name': entity_name, 'type': 'politician', 'position': politician_positions[idx]}
            focused_entity_position = politician_positions[idx]
            
            # Find the movement this politician belongs to
            movement_name = politician_movements[idx]
            if movement_name in movement_names:
                movement_idx = movement_names.index(movement_name)
                related_entities.append({
                    'name': movement_name,
                    'type': 'movement',
                    'position': movement_positions[movement_idx],
                    'relation': 'belongs_to'
                })
                
                # Find other politicians in the same movement
                for i, pol_movement in enumerate(politician_movements):
                    if pol_movement == movement_name and politician_names[i] != entity_name:
                        related_entities.append({
                            'name': politician_names[i],
                            'type': 'politician',
                            'position': politician_positions[i],
                            'relation': 'colleague'
                        })
        
        # If we found the focused entity, adjust the visualization
        if focused_entity_position is not None:
            # We'll keep track of which entities to include in the hybrid visualization
            include_movements = set()
            include_politicians = set()
            
            # Always include the focused entity
            if focused_entity['type'] == 'movement':
                include_movements.add(focused_entity['name'])
            else:
                include_politicians.add(focused_entity['name'])
                
            # Include related entities
            for entity in related_entities:
                if entity['type'] == 'movement':
                    include_movements.add(entity['name'])
                else:
                    include_politicians.add(entity['name'])
            
            # Filter movement and politician positions for hybrid view
            hybrid_movement_positions = []
            hybrid_movement_names = []
            hybrid_politician_positions = []
            hybrid_politician_names = []
            
            for i, name in enumerate(movement_names):
                if name in include_movements:
                    hybrid_movement_positions.append(movement_positions[i])
                    hybrid_movement_names.append(name)
                    
            for i, name in enumerate(politician_names):
                if name in include_politicians:
                    hybrid_politician_positions.append(politician_positions[i])
                    hybrid_politician_names.append(name)
            
            # Replace the original lists with filtered ones for hybrid mode
            if hybrid_mode:
                movement_positions = np.array(hybrid_movement_positions) if hybrid_movement_positions else np.array([])
                movement_names = hybrid_movement_names
                politician_positions = np.array(hybrid_politician_positions) if hybrid_politician_positions else np.array([])
                politician_names = hybrid_politician_names

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

    # Standardize the range for visualization
    max_range = np.linalg.norm(all_positions, axis=1).max()
    
    # Define standard scale for axes (-1.0 to 1.0)
    standard_max = 1.0
    scale_factor = standard_max / max_range
    
    # Scale positions to standard range
    movement_positions = movement_positions * scale_factor
    politician_positions = politician_positions * scale_factor
    all_positions = np.vstack((movement_positions, politician_positions))
    
    # Set max_range to our standard scale
    max_range = standard_max

    # Define colors for each dimension axis
    axis_colors = {
        'economic_axis': 'rgba(255, 165, 0, 0.8)',  # Orange
        'social_axis': 'rgba(65, 105, 225, 0.8)',    # Royal Blue
        'ecological_axis': 'rgba(50, 205, 50, 0.8)'  # Lime Green
    }
    
    # Define colors and opacity for reference planes
    plane_colors = {
        'xy': 'rgba(255, 165, 0, 0.1)',  # Orange (economic-social plane)
        'xz': 'rgba(50, 205, 50, 0.1)',   # Green (economic-ecological plane)
        'yz': 'rgba(65, 105, 225, 0.1)'    # Blue (social-ecological plane)
    }
    
    # Define fixed orthogonal axes for German political space
    # Instead of data-driven axes, we use fixed axes that better represent
    # the German political landscape
    axis_vectors = {}
    fixed_axes = {
        'economic_axis': np.array([1.0, 0.0, 0.0]),  # X axis: Market Liberal (+) to Social Democratic (-)
        'social_axis': np.array([0.0, 1.0, 0.0]),     # Y axis: Progressive (+) to Conservative (-)
        'ecological_axis': np.array([0.0, 0.0, 1.0])  # Z axis: Green (+) to Industrial (-)
    }
    
    # Define axis labels
    axis_labels = {
        'economic_axis': {'positive': 'Market Liberal', 'negative': 'Social Democratic'},
        'social_axis': {'positive': 'Progressive', 'negative': 'Conservative'},
        'ecological_axis': {'positive': 'Green', 'negative': 'Industrial'}
    }
    
    # Ensure the political dimensions match our fixed axes
    political_dimensions = ['economic_axis', 'social_axis', 'ecological_axis']
    
    for dim_name in axes_to_draw:
        if dim_name not in fixed_axes:
            continue
            
        unit_vector = fixed_axes[dim_name]
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
        
        # Add axis label at the end with specific German political context
        fig.add_trace(go.Scatter3d(
            x=[axis_end[0]], y=[axis_end[1]], z=[axis_end[2]],
            mode='text',
            text=[f"{axis_labels[dim_name]['positive']}"],
            textfont=dict(color=axis_color, size=12),
            hoverinfo='text', 
            hovertext=f"{dim_name.replace('_', ' ').title()}: {axis_labels[dim_name]['positive']}",
            showlegend=False
        ))
        
        # Add axis label at the negative end with specific German political context
        fig.add_trace(go.Scatter3d(
            x=[negative_end[0]], y=[negative_end[1]], z=[negative_end[2]],
            mode='text',
            text=[f"{axis_labels[dim_name]['negative']}"],
            textfont=dict(color=axis_color, size=12),
            hoverinfo='text',
            hovertext=f"{dim_name.replace('_', ' ').title()}: {axis_labels[dim_name]['negative']}",
            showlegend=False
        ))
        
        # Add tick marks along the axis in both positive and negative directions
        for tick_val in np.arange(0.2, 1.0, 0.2):
            # Standardized tick value (convert to -1.0 to 1.0 scale)
            std_tick_val = tick_val
            
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
                text=[f'+{std_tick_val:.1f}'],
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
                text=[f'-{std_tick_val:.1f}'],
                textfont=dict(color=axis_color, size=10),
                hoverinfo='none', showlegend=False
            ))

    # Reference planes removed as requested
    
    # Octant labels removed as requested
    
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
            # Convert to standardized scale (-1.0 to 1.0)
            std_score = score * 2 - 1
            hovertext += f"{dim.replace('_', ' ').title()}: {std_score:.2f}<br>"
        
        movement_hovertemplates.append(hovertext + "<extra></extra>")
    
    fig.add_trace(go.Scatter3d(
        x=movement_positions[:, 0], y=movement_positions[:, 1], z=movement_positions[:, 2],
        mode='markers+text', text=movement_names, textposition='top center',
        marker=dict(size=12, color=[movement_colors.get(m, 'grey') for m in movement_names], opacity=0.9, line=dict(color='white', width=1)),
        name='Movements', hovertemplate=movement_hovertemplates, hoverinfo='text'
    ))

    # --- 4. Add Word Clouds ---
    # Add word clouds for each movement positioned in the latent space
    for i, movement_name in enumerate(movement_names):
        movement_pos = movement_positions[i]
        movement_color = movement_colors.get(movement_name, 'grey')
        
        # Get word cloud for this movement
        movement_data = data['movements'][movement_name]
        word_cloud = movement_data.get('word_cloud', {})
        
        # If word cloud is empty, skip
        if not word_cloud:
            continue
            
        # Create word cloud points
        word_positions = []
        word_texts = []
        word_sizes = []
        word_colors = []
        word_hovertexts = []
        
        # Position each word based on its political dimension scores
        for word, weight in word_cloud.items():
            # Get dimension scores for this word
            if word.lower() in WORD_DIMENSION_SCORES:
                # Use predefined dimension scores
                dim_scores = WORD_DIMENSION_SCORES[word.lower()]
                
                # Calculate position based on dimension scores and movement position
                # Scale factor determines how far words are from their movement
                # Increase scale factor to spread words out more
                scale_factor = 0.25 * max_range * weight
                
                # Position is movement position + scaled dimension vector
                # Now we're explicitly mapping each dimension to its corresponding axis
                # Economic dimension (X), Social dimension (Y), Ecological dimension (Z)
                word_pos = [
                    movement_pos[0] + dim_scores[0] * scale_factor,  # Economic axis (X)
                    movement_pos[1] + dim_scores[1] * scale_factor,  # Social axis (Y)
                    movement_pos[2] + dim_scores[2] * scale_factor   # Ecological axis (Z)
                ]
                
                # Add jitter to prevent exact overlaps (increased jitter)
                jitter = np.random.normal(0, 0.02 * max_range, 3)
                word_pos = [word_pos[0] + jitter[0], word_pos[1] + jitter[1], word_pos[2] + jitter[2]]
                
                word_positions.append(word_pos)
                word_texts.append(word)
                
                # Size based on weight (importance in word cloud) - increased sizes
                word_sizes.append(5 + weight * 10)  # Scale between 5-15 based on weight
                
                # Color based on dimension scores - blend RGB values based on dimensions
                # Red = Economic (market liberal), Blue = Social (progressive), Green = Ecological (green)
                r = 0.5 + 0.5 * dim_scores[0]  # Economic: -1 to 1 → 0 to 1
                g = 0.5 + 0.5 * dim_scores[2]  # Ecological: -1 to 1 → 0 to 1
                b = 0.5 + 0.5 * dim_scores[1]  # Social: -1 to 1 → 0 to 1
                
                # Convert to hex color
                color = f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
                word_colors.append(color)
                
                # Create hover text with dimension information
                hovertext = f"<b>{word}</b> ({movement_name})<br>Weight: {weight:.2f}<br><br>"
                hovertext += "<b>Political Position:</b><br>"
                hovertext += f"Economic: {dim_scores[0]:.2f} ({'Market Liberal' if dim_scores[0] > 0 else 'Social Democratic'})<br>"
                hovertext += f"Social: {dim_scores[1]:.2f} ({'Progressive' if dim_scores[1] > 0 else 'Conservative'})<br>"
                hovertext += f"Ecological: {dim_scores[2]:.2f} ({'Green' if dim_scores[2] > 0 else 'Industrial'})"
                
                word_hovertexts.append(hovertext)
        
        # Add trace for this movement's word cloud
        if word_positions:
            # Convert to numpy arrays for easier indexing
            word_positions = np.array(word_positions)
            
            # Add word cloud points with individual colors
            fig.add_trace(go.Scatter3d(
                x=word_positions[:, 0], y=word_positions[:, 1], z=word_positions[:, 2],
                mode='markers',
                marker=dict(
                    size=word_sizes,
                    color=word_colors,  # Individual colors based on dimension scores
                    opacity=0.9,  # Increased opacity
                    symbol='diamond',  # Changed symbol for better visibility
                    line=dict(color='white', width=1.0)  # Increased line width
                ),
                text=word_texts,
                hoverinfo='text',
                hovertext=word_hovertexts,
                name=f"{movement_name} Word Cloud",
                showlegend=True  # Show in legend
            ))
            
            # Add text labels for ALL words to make them more visible
            fig.add_trace(go.Scatter3d(
                x=word_positions[:, 0],
                y=word_positions[:, 1],
                z=word_positions[:, 2],
                mode='text',
                text=word_texts,
                textposition="top center",
                textfont=dict(color='white', size=12),  # Increased text size
                hoverinfo='none',
                showlegend=False
            ))
            
            # Add connecting lines from movement to each word
            for j, word_pos in enumerate(word_positions):
                # Create a line from movement to word
                line_x = [movement_pos[0], word_pos[0]]
                line_y = [movement_pos[1], word_pos[1]]
                line_z = [movement_pos[2], word_pos[2]]
                
                fig.add_trace(go.Scatter3d(
                    x=line_x, y=line_y, z=line_z,
                    mode='lines',
                    line=dict(color=word_colors[j], width=2, dash='dot'),  # Dotted line with word's color
                    opacity=0.5,
                    hoverinfo='none',
                    showlegend=False
                ))
    
    # --- 5. Add Politicians (Moons) ---
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
            # Convert to standardized scale (-1.0 to 1.0)
            std_score = score * 2 - 1
            hovertext += f"{dim.replace('_', ' ').title()}: {std_score:.2f}<br>"
        
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

    # --- 6. Add Projection Lines for Selected Entity ---
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
            # Get entity dimension scores from the data
            entity_dim_scores = {}
            entity_data = None
            
            if entity_type == 'movement' and selected_name in data['movements']:
                entity_data = data['movements'][selected_name]
            elif entity_type == 'politician' and selected_name in data['politicians']:
                entity_data = data['politicians'][selected_name]
            
            if entity_data and 'position' in entity_data and 'expert_dimensions' in entity_data['position'] and 'axes' in entity_data['position']['expert_dimensions']:
                for dim_name in axes_to_draw:
                    # Get the standardized score (-1.0 to 1.0) for each dimension
                    raw_score = entity_data['position']['expert_dimensions']['axes'].get(dim_name, 0.5)
                    # Convert from [0,1] to [-1,1] scale
                    entity_dim_scores[dim_name] = raw_score * 2 - 1
            
            # Define fixed axis endpoints for projection lines
            # These are the actual axes in 3D space
            axis_endpoints = {
                'economic_axis': (np.array([1, 0, 0]) * max_range * 0.8, np.array([-1, 0, 0]) * max_range * 0.8),  # X-axis
                'social_axis': (np.array([0, 1, 0]) * max_range * 0.8, np.array([0, -1, 0]) * max_range * 0.8),    # Y-axis
                'ecological_axis': (np.array([0, 0, 1]) * max_range * 0.8, np.array([0, 0, -1]) * max_range * 0.8)  # Z-axis
            }
            
            # Create projection lines for each axis
            for axis_index, (axis_name, axis_endpoints_pair) in enumerate(axis_endpoints.items()):
                # Create a perpendicular offset that's consistent for each axis
                offset_vector = np.zeros(3)
                
                # Different offset strategy for each axis to ensure visibility
                if axis_index == 0:  # X-axis (Economic)
                    offset_vector[1] = 0.15 * max_range  # Offset in Y direction
                    offset_vector[2] = 0.05 * max_range  # Small offset in Z
                    axis_color = 'rgba(255, 0, 0, 0.8)'  # Red for economic
                elif axis_index == 1:  # Y-axis (Social)
                    offset_vector[0] = 0.15 * max_range  # Offset in X direction
                    offset_vector[2] = 0.05 * max_range  # Small offset in Z
                    axis_color = 'rgba(0, 0, 255, 0.8)'  # Blue for social
                else:  # Z-axis (Ecological)
                    offset_vector[0] = 0.05 * max_range  # Small offset in X
                    offset_vector[1] = 0.15 * max_range  # Offset in Y direction
                    axis_color = 'rgba(0, 255, 0, 0.8)'  # Green for ecological
                
                # Get the dimension score for this axis
                dim_name = axis_name.replace('_axis', '')
                score = entity_dim_scores.get(dim_name, 0)
                
                # Calculate projection point on this axis
                # The axis endpoints are at max_range * 0.8 in each direction
                # So we scale the score (-1 to 1) to get the projection point
                axis_vector = np.zeros(3)
                axis_vector[axis_index] = score * max_range * 0.8
                projection_point = axis_vector
                
                # Format the value to display
                display_value = f"{score:.2f}"
                
                # Add marker and text with improved visibility
                fig.add_trace(go.Scatter3d(
                    x=[projection_point[0] + offset_vector[0]], 
                    y=[projection_point[1] + offset_vector[1]], 
                    z=[projection_point[2] + offset_vector[2]],
                    mode='markers+text', 
                    marker=dict(size=10, color=axis_color, opacity=1.0),
                    text=[display_value],
                    textposition="middle center",
                    textfont=dict(color=axis_color, size=16, family="Arial Black"),
                    hoverinfo='text', 
                    hovertext=f"{dim_name.replace('_', ' ').title()}: {score:.2f}",
                    showlegend=False
                ))
                
                # Add a more visible marker at the exact projection point
                fig.add_trace(go.Scatter3d(
                    x=[projection_point[0]], 
                    y=[projection_point[1]], 
                    z=[projection_point[2]],
                    mode='markers', 
                    marker=dict(size=8, color=axis_color, symbol='diamond', opacity=1.0),
                    hoverinfo='text', 
                    hovertext=f"{dim_name.replace('_', ' ').title()}: {score:.2f}",
                    showlegend=False
                ))
                
                # Draw line from origin to projection point
                fig.add_trace(go.Scatter3d(
                    x=[0, projection_point[0]], 
                    y=[0, projection_point[1]], 
                    z=[0, projection_point[2]],
                    mode='lines', 
                    line=dict(color=axis_color, width=4, dash='dot'),
                    hoverinfo='none',
                    showlegend=False
                ))
    
    # --- Layout and Final Touches ---
    # Camera distance from center
    camera_distance = 1.8
    
    # Legend text removed as requested
                 
    # Set up the 3D layout with dark theme - no title for clean full-screen view
    fig.update_layout(
        template='plotly_dark',
        scene=dict(
            xaxis_title='Economic Axis', 
            yaxis_title='Social Axis', 
            zaxis_title='Ecological Axis',
            xaxis=dict(showbackground=False, gridcolor='#444', zerolinecolor='#444'),
            yaxis=dict(showbackground=False, gridcolor='#444', zerolinecolor='#444'),
            zaxis=dict(showbackground=False, gridcolor='#444', zerolinecolor='#444'),
            bgcolor='rgb(10,10,25)',
            # Since data is centered at origin, camera naturally rotates around center of mass
            camera=dict(
                eye=dict(x=camera_distance, y=camera_distance, z=camera_distance),
                center=dict(x=0, y=0, z=0)  # Look at origin where center of mass is now located
            ),
            annotations=[]  # Removed legend annotation
        ),
        paper_bgcolor='rgb(10,10,25)',
        plot_bgcolor='rgb(10,10,25)',
        margin=dict(l=0, r=0, t=0, b=0),  # Removed top margin (t=0) for full-screen view
        legend=dict(
            font=dict(color='white', size=10),
            bgcolor='rgba(30,30,50,0.8)',
            bordercolor='#444',
            borderwidth=1,
            itemsizing='constant',
            yanchor="top", 
            y=0.99, 
            xanchor="left", 
            x=0.01
        )
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
    
    # Parse selected entity if provided
    selected_entity = None
    if args.select:
        try:
            entity_type, entity_name = args.select.split(':', 1)
            selected_entity = {'type': entity_type, 'name': entity_name}
            print(f"Showing projections for {entity_type}: {entity_name}")
        except ValueError:
            print(f"Invalid format for --select. Use 'type:name', e.g. 'movement:cdu'")
    
    # Create visualization
    print("Creating 3D galaxy visualization...")
    create_galaxy_visualization(data, args.output, args.seed, selected_entity)
    
    print("Galaxy visualization complete.")

if __name__ == '__main__':
    main()
