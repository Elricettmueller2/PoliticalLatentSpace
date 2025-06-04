#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latent Space Explorer
--------------------
Visualizes political movements in latent space with interpretable dimensions
"""

import pandas as pd
import numpy as np
import re
import spacy
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

# Extract key terms for each movement
print("Extracting key terms for each movement...")

# Load German language model for NLP
try:
    nlp = spacy.load("de_core_news_sm")
    print("Loaded German language model")
except:
    print("German language model not found, installing...")
    import os
    os.system("python -m spacy download de_core_news_sm")
    nlp = spacy.load("de_core_news_sm")

# Function to preprocess text and extract important terms
def extract_key_terms(text, top_n=20):
    # Process with spaCy
    doc = nlp(text)
    
    # Extract nouns, proper nouns, and adjectives
    important_tokens = [token.lemma_ for token in doc 
                      if (token.pos_ in ["NOUN", "PROPN", "ADJ"]) 
                      and not token.is_stop
                      and len(token.text) > 3]
    
    # Count frequencies
    term_freq = Counter(important_tokens)
    
    # Return top N terms
    return term_freq.most_common(top_n)

# Extract key terms for each movement
df['key_terms'] = df['text'].apply(extract_key_terms)

# Create TF-IDF vectorizer to find distinctive terms
print("Identifying distinctive terms...")
tfidf = TfidfVectorizer(max_features=1000, stop_words=nlp.Defaults.stop_words)
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
                for idx in sorted_indices[:10] if tfidf_scores[idx] > 0]
    df.at[i, 'distinctive_terms'] = top_terms

# Create semantic dimensions based on key political concepts
print("Creating semantic dimensions...")
political_concepts = {
    "Umwelt": ["Klima", "Umwelt", "nachhaltig", "ökologisch", "erneuerbar", "Energie"],
    "Wirtschaft": ["Wirtschaft", "Markt", "Unternehmen", "Wachstum", "Wettbewerb", "Steuer"],
    "Soziales": ["sozial", "Gerechtigkeit", "Solidarität", "Armut", "Gleichheit", "Teilhabe"],
    "Migration": ["Migration", "Flüchtling", "Integration", "Asyl", "Einwanderung", "Grenzen"],
    "Europa": ["Europa", "EU", "europäisch", "Integration", "Brüssel", "Gemeinschaft"],
    "Tradition": ["Tradition", "Familie", "Werte", "konservativ", "Heimat", "Identität"],
    "Demokratie": ["Demokratie", "Bürger", "Parlament", "Mitbestimmung", "Freiheit", "Recht"]
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

# Create interactive visualization with concept explanations
print("Creating interactive visualization...")
fig = go.Figure()

# Add scatter points for movements
for i, row in df.iterrows():
    # Format distinctive terms for hover text
    distinctive_html = "<br>".join([f"<b>{term}</b>: {score:.3f}" 
                                  for term, score in row['distinctive_terms']])
    
    # Format concept scores for hover text
    concept_scores = row['concept_scores']
    # Sort concepts by score
    sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
    concept_html = "<br>".join([f"<b>{concept}</b>: {score:.1f}" 
                              for concept, score in sorted_concepts])
    
    # Create hover text with all information
    hover_text = f"""
    <b>{row['movement']}</b><br>
    <br>
    <b>Politische Dimensionen:</b><br>
    {concept_html}<br>
    <br>
    <b>Charakteristische Begriffe:</b><br>
    {distinctive_html}<br>
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

# Update layout
fig.update_layout(
    title="Politischer Latent Space mit interpretierbaren Dimensionen",
    xaxis=dict(
        title="Latent Dimension 1",
        showgrid=True,
        gridcolor='rgba(200,200,200,0.2)',
        zeroline=False
    ),
    yaxis=dict(
        title="Latent Dimension 2",
        showgrid=True,
        gridcolor='rgba(200,200,200,0.2)',
        zeroline=False
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

# Create a version with semantic axes
print("Creating semantic axes visualization...")

# Calculate movement positions in semantic space
# We'll use two key dimensions: Economic (left-right) and Social (progressive-conservative)
economic_terms = political_concepts["Wirtschaft"] + ["privat", "Staat", "Regulierung", "Freiheit", "Markt"]
social_terms = political_concepts["Tradition"] + political_concepts["Migration"] + ["progressiv", "konservativ"]

# Function to calculate position on semantic axis
def calculate_axis_position(text, axis_terms):
    text_lower = text.lower()
    score = sum(text_lower.count(term.lower()) for term in axis_terms)
    return score / (len(text.split()) + 1) * 1000

# Calculate semantic positions
df['economic_axis'] = df['text'].apply(lambda x: calculate_axis_position(x, economic_terms))
df['social_axis'] = df['text'].apply(lambda x: calculate_axis_position(x, social_terms))

# Create semantic space visualization
fig_semantic = go.Figure()

# Add scatter points for movements in semantic space
for i, row in df.iterrows():
    # Use the same hover text as before
    distinctive_html = "<br>".join([f"<b>{term}</b>: {score:.3f}" 
                                  for term, score in row['distinctive_terms']])
    
    concept_scores = row['concept_scores']
    sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
    concept_html = "<br>".join([f"<b>{concept}</b>: {score:.1f}" 
                              for concept, score in sorted_concepts])
    
    hover_text = f"""
    <b>{row['movement']}</b><br>
    <br>
    <b>Politische Dimensionen:</b><br>
    {concept_html}<br>
    <br>
    <b>Charakteristische Begriffe:</b><br>
    {distinctive_html}<br>
    """
    
    fig_semantic.add_trace(go.Scatter(
        x=[row['economic_axis']],
        y=[row['social_axis']],
        mode='markers+text',
        marker=dict(size=15, color=row['color']),
        text=row['movement'],
        textposition="top center",
        name=row['movement'],
        hovertext=hover_text,
        hoverinfo="text"
    ))

# Update layout for semantic space
fig_semantic.update_layout(
    title="Politischer Raum mit semantischen Achsen",
    xaxis=dict(
        title="Wirtschaftliche Dimension",
        showgrid=True,
        gridcolor='rgba(200,200,200,0.2)',
        zeroline=True
    ),
    yaxis=dict(
        title="Gesellschaftliche Dimension",
        showgrid=True,
        gridcolor='rgba(200,200,200,0.2)',
        zeroline=True
    ),
    plot_bgcolor='rgba(240,240,240,0.8)',
    hovermode='closest',
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial"
    )
)

# Save semantic space visualization
fig_semantic.write_html("semantic_space_explorer.html")
print("Saved semantic space visualization to semantic_space_explorer.html")

print("\nDone! You can now explore the latent space with interpretable dimensions.")
