#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Political Landscape Visualization
--------------------------------
Creates a meaningful political landscape visualization using metaphors
people can intuitively understand.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import random
import colorsys

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

# Calculate similarity matrix
similarity_matrix = cosine_similarity(embeddings)
np.fill_diagonal(similarity_matrix, 0)  # Remove self-similarity

# Use PCA for dimensionality reduction (more interpretable axes than UMAP)
print("Performing PCA for interpretable dimensions...")
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)
df['x'] = reduced_embeddings[:, 0]
df['y'] = reduced_embeddings[:, 1]

# Map movement types and colors
df['movement_type'] = df['movement'].map(movement_types)
df['color'] = df['movement_type'].map(color_map)

# Create hierarchical clustering
print("Creating hierarchical clustering...")
Z = linkage(embeddings, 'ward')

# Define key political axes
# We'll use the PCA components to interpret meaningful political dimensions
print("Analyzing political dimensions...")

# Extract the top terms that contribute to each PCA component
# This is a simplified approach - in reality we would use more sophisticated methods
component_terms = {
    "x_axis": "Links vs. Rechts (Wirtschaftspolitik, Gesellschaftsbild)",
    "y_axis": "Progressiv vs. Konservativ (Werte, Traditionen)"
}

# Create a political landscape visualization
print("Creating political landscape visualization...")

# Function to create a more appealing color for territories
def get_territory_color(base_color, alpha=0.3):
    # Convert hex to RGB
    if base_color.startswith('#'):
        rgb = mcolors.hex2color(base_color)
    else:
        rgb = mcolors.to_rgb(base_color)
    
    # Convert to HSL, lighten, and convert back to RGB
    h, l, s = colorsys.rgb_to_hls(*rgb)
    l = min(1.0, l * 1.5)  # Lighten
    s = max(0.0, s * 0.7)  # Reduce saturation
    rgb = colorsys.hls_to_rgb(h, l, s)
    
    # Return with alpha
    return (*rgb, alpha)

# Create a Voronoi diagram to represent political territories
points = df[['x', 'y']].values
vor = Voronoi(points)

# Create a figure with a political map aesthetic
fig, ax = plt.subplots(figsize=(14, 10))

# Plot Voronoi regions as territories
for i, region in enumerate(vor.regions):
    if not -1 in region and len(region) > 0:
        polygon = [vor.vertices[i] for i in region]
        if len(polygon) > 2:  # Need at least 3 points for a polygon
            # Find the closest point to this region
            distances = []
            for j, point in enumerate(points):
                # Calculate distance to the center of the polygon
                center = np.mean(polygon, axis=0)
                dist = np.linalg.norm(point - center)
                distances.append((dist, j))
            
            closest_idx = min(distances)[1]
            movement_color = df.iloc[closest_idx]['color']
            territory_color = get_territory_color(movement_color)
            
            ax.fill(*zip(*polygon), color=territory_color, edgecolor='gray', 
                   linewidth=1, alpha=0.5)

# Add movement points
for i, row in df.iterrows():
    ax.scatter(row['x'], row['y'], color=row['color'], s=100, zorder=5, 
              edgecolor='black', linewidth=1)
    
    # Add movement labels
    ax.annotate(row['movement'], (row['x'], row['y']), 
               xytext=(5, 5), textcoords='offset points',
               fontsize=11, fontweight='bold', zorder=6)

# Add political compass axes
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)

# Label the axes with political dimensions
ax.set_xlabel(component_terms['x_axis'], fontsize=14)
ax.set_ylabel(component_terms['y_axis'], fontsize=14)

# Add quadrant labels
ax.text(max(df['x'])*0.8, max(df['y'])*0.8, "Progressiv-Links", 
        fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.5))
ax.text(max(df['x'])*0.8, min(df['y'])*0.8, "Progressiv-Rechts", 
        fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.5))
ax.text(min(df['x'])*0.8, max(df['y'])*0.8, "Konservativ-Links", 
        fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.5))
ax.text(min(df['x'])*0.8, min(df['y'])*0.8, "Konservativ-Rechts", 
        fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.5))

# Add a title and description
plt.title("Die Deutsche Politische Landschaft", fontsize=18)
plt.figtext(0.5, 0.01, 
           "Basierend auf der semantischen Analyse von Partei- und Bewegungstexten.\n"
           "Ähnliche Positionen liegen näher beieinander, die Territorien zeigen Einflussbereiche.", 
           ha="center", fontsize=10)

# Add a legend for movement types
handles = []
labels = []
for movement_type, color in color_map.items():
    if movement_type in df['movement_type'].values:
        handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=10))
        # Convert movement_type to a more readable format
        readable_type = movement_type.replace('_', ' ').title()
        labels.append(readable_type)

ax.legend(handles, labels, loc='upper right', title="Bewegungstypen")

# Style the plot to look like a map
ax.set_facecolor('#f0f0f0')  # Light gray background
plt.grid(True, linestyle='--', alpha=0.3)

# Remove ticks but keep the axes
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Save the political landscape visualization
plt.tight_layout()
plt.savefig('political_landscape.png', dpi=300, bbox_inches='tight')
print("Saved political landscape to political_landscape.png")

# Create an interactive version with Plotly
print("Creating interactive political landscape...")
fig = go.Figure()

# Add scatter points for movements
for i, row in df.iterrows():
    fig.add_trace(go.Scatter(
        x=[row['x']],
        y=[row['y']],
        mode='markers+text',
        marker=dict(size=15, color=row['color'], line=dict(width=1, color='black')),
        text=row['movement'],
        textposition="top center",
        name=row['movement'],
        hovertext=f"<b>{row['movement']}</b><br><br>{row['text'][:200]}...",
        hoverinfo="text"
    ))

# Add political compass axes
fig.add_shape(
    type="line",
    x0=min(df['x'])*1.2, y0=0,
    x1=max(df['x'])*1.2, y1=0,
    line=dict(color="gray", width=1, dash="dash")
)
fig.add_shape(
    type="line",
    x0=0, y0=min(df['y'])*1.2,
    x1=0, y1=max(df['y'])*1.2,
    line=dict(color="gray", width=1, dash="dash")
)

# Add quadrant labels
fig.add_annotation(x=max(df['x'])*0.8, y=max(df['y'])*0.8, text="Progressiv-Links",
                  showarrow=False, bgcolor="rgba(255,255,255,0.5)")
fig.add_annotation(x=max(df['x'])*0.8, y=min(df['y'])*0.8, text="Progressiv-Rechts",
                  showarrow=False, bgcolor="rgba(255,255,255,0.5)")
fig.add_annotation(x=min(df['x'])*0.8, y=max(df['y'])*0.8, text="Konservativ-Links",
                  showarrow=False, bgcolor="rgba(255,255,255,0.5)")
fig.add_annotation(x=min(df['x'])*0.8, y=min(df['y'])*0.8, text="Konservativ-Rechts",
                  showarrow=False, bgcolor="rgba(255,255,255,0.5)")

# Update layout
fig.update_layout(
    title="Die Deutsche Politische Landschaft (Interaktiv)",
    xaxis=dict(
        title=component_terms['x_axis'],
        showgrid=True,
        gridcolor='rgba(200,200,200,0.2)',
        showticklabels=False,
        zeroline=False
    ),
    yaxis=dict(
        title=component_terms['y_axis'],
        showgrid=True,
        gridcolor='rgba(200,200,200,0.2)',
        showticklabels=False,
        zeroline=False
    ),
    plot_bgcolor='#f0f0f0',
    hovermode='closest',
    legend_title="Bewegungstypen",
    annotations=[
        dict(
            text="Basierend auf der semantischen Analyse von Partei- und Bewegungstexten.<br>Ähnliche Positionen liegen näher beieinander.",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=-0.1,
            font=dict(size=10)
        )
    ]
)

# Save the interactive visualization
fig.write_html("political_landscape_interactive.html")
print("Saved interactive political landscape to political_landscape_interactive.html")

print("\nDone! You can now explore the political landscape visualizations.")
