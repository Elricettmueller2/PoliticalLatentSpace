#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vector Space Interpreter
-----------------------
Creates a visualization that shows what makes each political movement
stand where it stands in the latent space.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the same data as in the main prototype
from political_space_prototype import sample_movements, movement_types, color_map

print("Loading sentence transformer model...")
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

# Create lists for easier processing
movements = list(sample_movements.keys())
texts = list(sample_movements.values())
colors = [color_map.get(movement_types.get(m, "other"), "#888888") for m in movements]

print(f"Loaded {len(movements)} political movements")

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(texts)

# Dimensionality reduction with UMAP
print("Performing dimensionality reduction...")
reducer = umap.UMAP(n_components=2, n_neighbors=4, min_dist=0.3, random_state=42)
reduced_embeddings = reducer.fit_transform(embeddings)
x_coords = reduced_embeddings[:, 0]
y_coords = reduced_embeddings[:, 1]

# Simple German stopwords list
german_stopwords = [
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "eines", 
    "einem", "einen", "und", "oder", "aber", "wenn", "dann", "als", "wie", "wo", 
    "wer", "was", "warum", "wieso", "weshalb", "welche", "welcher", "welches", 
    "für", "von", "mit", "zu", "zur", "zum", "auf", "in", "im", "bei", "an", 
    "am", "um", "durch", "über", "unter", "gegen", "nach", "vor", "ist", "sind", 
    "war", "waren", "wird", "werden", "wurde", "wurden", "hat", "haben", "hatte", 
    "hatten", "kann", "können", "darf", "dürfen", "muss", "müssen", "soll", 
    "sollen", "will", "wollen", "mag", "mögen", "dass", "daß", "weil", "obwohl", 
    "damit", "dafür", "dabei", "dazu", "daran", "darauf", "darunter", "darüber",
    "ich", "du", "er", "sie", "es", "wir", "ihr", "sie", "mich", "dich", "sich",
    "uns", "euch", "mir", "dir", "ihm", "ihr", "ihnen", "mein", "dein", "sein",
    "unser", "euer", "ihr", "nicht", "auch", "nur", "schon", "noch", "wieder",
    "immer", "alle", "alles", "jeder", "jede", "jedes", "man", "selbst", "so"
]

# Extract distinctive terms using TF-IDF
print("Extracting distinctive terms...")
tfidf = TfidfVectorizer(max_features=1000, stop_words=german_stopwords)
tfidf_matrix = tfidf.fit_transform(texts)
feature_names = np.array(tfidf.get_feature_names_out())

# Extract top terms for each movement
distinctive_terms = []
for i in range(len(movements)):
    # Get the TF-IDF scores for this document
    tfidf_scores = tfidf_matrix[i].toarray()[0]
    # Sort terms by TF-IDF score
    sorted_indices = np.argsort(tfidf_scores)[::-1]
    # Get top terms
    top_terms = [(feature_names[idx], tfidf_scores[idx]) 
                for idx in sorted_indices[:10] if tfidf_scores[idx] > 0]
    distinctive_terms.append(top_terms)

# Find nearest neighbors for each movement
nearest_neighbors = []
for i, emb in enumerate(embeddings):
    # Calculate similarities to all other embeddings
    similarities = cosine_similarity([emb], embeddings)[0]
    # Sort by similarity (excluding self)
    sorted_indices = np.argsort(similarities)[::-1][1:4]  # Top 3 neighbors
    neighbor_info = [(movements[idx], similarities[idx]) for idx in sorted_indices]
    nearest_neighbors.append(neighbor_info)

# Create interactive visualization
print("Creating interactive visualization...")
fig = go.Figure()

# Add scatter points for movements
for i in range(len(movements)):
    # Format distinctive terms for hover text
    terms_html = "<br>".join([f"<b>{term}</b>: {score:.3f}" 
                            for term, score in distinctive_terms[i]])
    
    # Format nearest neighbors
    neighbors_html = "<br>".join([f"<b>{neighbor}</b>: {sim:.3f}" 
                                for neighbor, sim in nearest_neighbors[i]])
    
    # Create hover text with all information
    hover_text = f"""
    <b>{movements[i]}</b><br>
    <br>
    <b>Charakteristische Begriffe:</b><br>
    {terms_html}<br>
    <br>
    <b>Ähnlichste Bewegungen:</b><br>
    {neighbors_html}<br>
    <br>
    <b>Position bestimmt durch:</b><br>
    • Thematische Schwerpunkte<br>
    • Verwendete Schlüsselbegriffe<br>
    • Semantische Ähnlichkeit zu anderen Bewegungen
    """
    
    fig.add_trace(go.Scatter(
        x=[x_coords[i]],
        y=[y_coords[i]],
        mode='markers+text',
        marker=dict(size=15, color=colors[i]),
        text=movements[i],
        textposition="top center",
        name=movements[i],
        hovertext=hover_text,
        hoverinfo="text"
    ))

# Add edges between similar movements
similarity_threshold = 0.7
for i in range(len(movements)):
    for j in range(i+1, len(movements)):
        # Calculate similarity between embeddings
        sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
        if sim > similarity_threshold:
            fig.add_trace(go.Scatter(
                x=[x_coords[i], x_coords[j]],
                y=[y_coords[i], y_coords[j]],
                mode='lines',
                line=dict(width=sim*3, color='rgba(120, 120, 120, 0.5)'),
                hoverinfo='text',
                hovertext=f"Similarity: {sim:.3f}",
                showlegend=False
            ))

# Update layout
fig.update_layout(
    title="Politischer Vektorraum mit Erklärungen",
    xaxis=dict(
        title="Latent Dimension 1",
        showgrid=True,
        gridcolor='rgba(200,200,200,0.2)',
        zeroline=False,
        showticklabels=False
    ),
    yaxis=dict(
        title="Latent Dimension 2",
        showgrid=True,
        gridcolor='rgba(200,200,200,0.2)',
        zeroline=False,
        showticklabels=False
    ),
    plot_bgcolor='rgba(240,240,240,0.8)',
    hovermode='closest',
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial"
    )
)

# Save interactive visualization
fig.write_html("vector_space_interpreter.html")
print("Saved interactive visualization to vector_space_interpreter.html")

print("\nDone! You can now explore the vector space with interpretable dimensions.")
