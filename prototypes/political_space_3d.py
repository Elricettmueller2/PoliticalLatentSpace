#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Political Movement Space Visualization
----------------------------------------
Visualizes political movements in a 3D interactive space
"""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# Load the same data as in the main prototype
from political_space_prototype import sample_movements, movement_types, color_map

# Create a DataFrame
df = pd.DataFrame(list(sample_movements.items()), columns=['movement', 'text'])
print(f"Loaded {len(df)} political movements")

# Load model and generate embeddings
print("Loading sentence transformer model...")
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
embeddings = model.encode(df['text'].tolist())
df['embedding'] = list(embeddings)

# 3D dimensionality reduction with UMAP
print("Performing 3D dimensionality reduction...")
reducer_3d = umap.UMAP(n_components=3, n_neighbors=4, min_dist=0.3, random_state=42)
reduced_embeddings_3d = reducer_3d.fit_transform(np.array([e for e in embeddings]))
df['x'] = reduced_embeddings_3d[:, 0]
df['y'] = reduced_embeddings_3d[:, 1]
df['z'] = reduced_embeddings_3d[:, 2]

# Calculate similarity matrix
similarity_matrix = cosine_similarity(embeddings)
np.fill_diagonal(similarity_matrix, 0)  # Remove self-similarity

# Map movement types and colors
df['movement_type'] = df['movement'].map(movement_types)
df['color'] = df['movement_type'].map(color_map)

# Create 3D visualization
fig = go.Figure()

# Add nodes (movements)
for i, row in df.iterrows():
    fig.add_trace(go.Scatter3d(
        x=[row['x']],
        y=[row['y']],
        z=[row['z']],
        mode='markers+text',
        marker=dict(
            size=15,
            color=row['color'],
            opacity=0.8
        ),
        text=row['movement'],
        name=row['movement'],
        textposition="top center",
        hoverinfo="text",
        hovertext=f"{row['movement']}<br>{row['text'][:100]}..."
    ))

# Add edges (connections) based on similarity threshold
threshold = 0.5
for i in range(len(df)):
    for j in range(i+1, len(df)):
        sim = similarity_matrix[i, j]
        if sim > threshold:
            # Get coordinates
            x0, y0, z0 = df.iloc[i]['x'], df.iloc[i]['y'], df.iloc[i]['z']
            x1, y1, z1 = df.iloc[j]['x'], df.iloc[j]['y'], df.iloc[j]['z']
            
            # Add line
            fig.add_trace(go.Scatter3d(
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                mode='lines',
                line=dict(
                    width=sim*5,
                    color='rgba(120, 120, 120, 0.5)'
                ),
                hoverinfo='text',
                hovertext=f"Similarity: {sim:.2f}",
                showlegend=False
            ))

# Update layout for better 3D visualization
fig.update_layout(
    title="3D Political Movement Space",
    scene=dict(
        xaxis=dict(showticklabels=False, title=''),
        yaxis=dict(showticklabels=False, title=''),
        zaxis=dict(showticklabels=False, title=''),
        bgcolor='rgba(240, 240, 240, 0.8)'
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    hovermode='closest',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

# Save interactive 3D visualization
fig.write_html("political_movement_3d.html")
print("Saved 3D visualization to political_movement_3d.html")
