#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latent Space Explorer (Simplified Version)
-----------------------------------------
Visualizes political movements in latent space with interpretable dimensions
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
from sentence_transformers import SentenceTransformer
import umap
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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

# Dimensionality reduction with UMAP
print("Performing dimensionality reduction...")
reducer = umap.UMAP(n_components=2, n_neighbors=4, min_dist=0.3, random_state=42)
reduced_embeddings = reducer.fit_transform(np.array([e for e in embeddings]))
df['x'] = reduced_embeddings[:, 0]
df['y'] = reduced_embeddings[:, 1]

# Map movement types and colors
df['movement_type'] = df['movement'].map(movement_types)
df['color'] = df['movement_type'].map(color_map)

# Extract key terms for each movement using TF-IDF
print("Extracting key terms using TF-IDF...")

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

# Create TF-IDF vectorizer to find distinctive terms
tfidf = TfidfVectorizer(max_features=1000, stop_words=german_stopwords)
tfidf_matrix = tfidf.fit_transform(df['text'])
feature_names = np.array(tfidf.get_feature_names_out())

# For each movement, find the most distinctive terms based on TF-IDF
df['distinctive_terms'] = None
for i, row in df.iterrows():
    # Get the TF-IDF scores for this document
    tfidf_scores = tfidf_matrix[i].toarray()[0]
    # Sort terms by TF-IDF score
    sorted_indices = np.argsort(tfidf_scores)[::-1]
    # Get top 10 terms
    top_terms = [(feature_names[idx], tfidf_scores[idx]) 
                for idx in sorted_indices[:15] if tfidf_scores[idx] > 0]
    df.at[i, 'distinctive_terms'] = top_terms

# Create semantic dimensions based on key political concepts
print("Creating semantic dimensions...")
political_concepts = {
    "Umwelt": ["klima", "umwelt", "nachhaltig", "ökologisch", "erneuerbar", "energie", "natur", "grün"],
    "Wirtschaft": ["wirtschaft", "markt", "unternehmen", "wachstum", "wettbewerb", "steuer", "finanzen", "handel"],
    "Soziales": ["sozial", "gerechtigkeit", "solidarität", "armut", "gleichheit", "teilhabe", "gemeinschaft"],
    "Migration": ["migration", "flüchtling", "integration", "asyl", "einwanderung", "grenzen", "ausländer"],
    "Europa": ["europa", "eu", "europäisch", "integration", "brüssel", "gemeinschaft", "union"],
    "Tradition": ["tradition", "familie", "werte", "konservativ", "heimat", "identität", "kultur"],
    "Demokratie": ["demokratie", "bürger", "parlament", "mitbestimmung", "freiheit", "recht", "verfassung"]
}

# Function to calculate concept score for a text
def calculate_concept_scores(text):
    scores = {}
    text_lower = text.lower()
    
    for concept, keywords in political_concepts.items():
        # Count occurrences of each keyword
        score = sum(text_lower.count(keyword.lower()) for keyword in keywords)
        # Normalize by text length
        scores[concept] = score / (len(text.split()) + 1) * 1000
    
    return scores

# Calculate concept scores for each movement
df['concept_scores'] = df['text'].apply(calculate_concept_scores)

# Calculate vector contribution to position
print("Calculating vector contributions to positions...")

# Function to find nearest neighbors in embedding space
def find_nearest_neighbors(embedding, all_embeddings, n=3):
    similarities = cosine_similarity([embedding], all_embeddings)[0]
    # Get indices of top n similar embeddings (excluding self)
    sorted_indices = np.argsort(similarities)[::-1]
    return sorted_indices[1:n+1], similarities[sorted_indices[1:n+1]]

# Find nearest neighbors for each movement
df['nearest_neighbors'] = None
for i, row in df.iterrows():
    neighbors, similarities = find_nearest_neighbors(row['embedding'], embeddings)
    neighbor_info = [(df.iloc[idx]['movement'], sim) for idx, sim in zip(neighbors, similarities)]
    df.at[i, 'nearest_neighbors'] = neighbor_info

# Create interactive visualization with concept explanations
print("Creating interactive visualization...")
fig = go.Figure()

# Add scatter points for movements
for i, row in df.iterrows():
    # Format distinctive terms for hover text
    distinctive_html = "<br>".join([f"<b>{term}</b>: {score:.3f}" 
                                  for term, score in row['distinctive_terms'][:10]])
    
    # Format concept scores for hover text
    concept_scores = row['concept_scores']
    # Sort concepts by score
    sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
    concept_html = "<br>".join([f"<b>{concept}</b>: {score:.1f}" 
                              for concept, score in sorted_concepts])
    
    # Format nearest neighbors
    neighbors_html = "<br>".join([f"<b>{neighbor}</b>: {sim:.3f}" 
                                for neighbor, sim in row['nearest_neighbors']])
    
    # Create hover text with all information
    hover_text = f"""
    <b>{row['movement']}</b><br>
    <br>
    <b>Politische Dimensionen:</b><br>
    {concept_html}<br>
    <br>
    <b>Charakteristische Begriffe:</b><br>
    {distinctive_html}<br>
    <br>
    <b>Ähnlichste Bewegungen:</b><br>
    {neighbors_html}
    """
    
    fig.add_trace(go.Scatter(
        x=[row['x']],
        y=[row['y']],
        mode='markers+text',
        marker=dict(size=15, color=row['color']),
        text=row['movement'],
        textposition="top center",
        name=row['movement'],
        hovertext=hover_text,
        hoverinfo="text"
    ))

# Add edges between similar movements
similarity_threshold = 0.7
for i, row_i in df.iterrows():
    for j, row_j in df.iterrows():
        if i < j:  # Avoid duplicate edges
            # Calculate similarity between embeddings
            sim = cosine_similarity([row_i['embedding']], [row_j['embedding']])[0][0]
            if sim > similarity_threshold:
                fig.add_trace(go.Scatter(
                    x=[row_i['x'], row_j['x']],
                    y=[row_i['y'], row_j['y']],
                    mode='lines',
                    line=dict(width=sim*3, color='rgba(120, 120, 120, 0.5)'),
                    hoverinfo='text',
                    hovertext=f"Similarity: {sim:.3f}",
                    showlegend=False
                ))

# Update layout
fig.update_layout(
    title="Politischer Latent Space mit interpretierbaren Dimensionen",
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
fig.write_html("latent_space_explorer.html")
print("Saved interactive visualization to latent_space_explorer.html")

print("\nDone! You can now explore the latent space with interpretable dimensions.")
