#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topic-Based Political Space Projection
-------------------------------------
Projects political movements onto specific topic dimensions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# Load the same data as in the main prototype
from political_space_prototype import sample_movements, movement_types, color_map

# Define key political topics
topics = {
    "climate": "Der Klimawandel ist eine existenzielle Bedrohung. Wir müssen sofort handeln, um die Treibhausgasemissionen zu reduzieren und auf erneuerbare Energien umsteigen.",
    "economy": "Die Wirtschaft muss wachsen und wettbewerbsfähig bleiben. Unternehmen brauchen gute Rahmenbedingungen und niedrige Steuern.",
    "migration": "Migration bereichert unsere Gesellschaft. Wir brauchen humane Asylverfahren und legale Einwanderungswege.",
    "security": "Die innere und äußere Sicherheit muss gestärkt werden. Wir brauchen eine gut ausgestattete Polizei und Bundeswehr.",
    "social": "Der Sozialstaat muss ausgebaut werden. Alle Menschen haben ein Recht auf soziale Absicherung und Teilhabe.",
    "europe": "Die europäische Integration muss vertieft werden. Wir brauchen mehr europäische Zusammenarbeit und Solidarität."
}

# Create a DataFrame
df = pd.DataFrame(list(sample_movements.items()), columns=['movement', 'text'])
print(f"Loaded {len(df)} political movements")

# Load model and generate embeddings
print("Loading sentence transformer model...")
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

# Generate embeddings for movements
movement_embeddings = model.encode(df['text'].tolist())
df['embedding'] = list(movement_embeddings)

# Generate embeddings for topics
topic_embeddings = {}
for topic_name, topic_text in topics.items():
    topic_embeddings[topic_name] = model.encode(topic_text)

# Calculate similarity of each movement to each topic
print("Calculating topic affinities...")
topic_affinities = {}
for topic_name, topic_embedding in topic_embeddings.items():
    similarities = []
    for movement_embedding in movement_embeddings:
        # Calculate cosine similarity
        similarity = cosine_similarity([movement_embedding], [topic_embedding])[0][0]
        similarities.append(similarity)
    df[f'topic_{topic_name}'] = similarities
    topic_affinities[topic_name] = similarities

# Map movement types and colors
df['movement_type'] = df['movement'].map(movement_types)
df['color'] = df['movement_type'].map(color_map)

# Create a radar chart for each movement
print("Creating radar charts...")
fig = go.Figure()

for i, row in df.iterrows():
    # Get topic values for this movement
    topic_values = [row[f'topic_{topic}'] for topic in topics.keys()]
    # Add a trace for this movement
    fig.add_trace(go.Scatterpolar(
        r=topic_values,
        theta=list(topics.keys()),
        fill='toself',
        name=row['movement'],
        line=dict(color=row['color']),
        opacity=0.7
    ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )
    ),
    title="Topic Affinities of Political Movements",
    showlegend=True
)

# Save interactive radar chart
fig.write_html("political_movement_topics.html")
print("Saved topic radar chart to political_movement_topics.html")

# Create a 2D projection using two selected topics
print("Creating 2D topic projections...")

# Function to create 2D topic projections
def create_topic_projection(topic1, topic2):
    fig = go.Figure()
    
    # Add scatter points for each movement
    for i, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row[f'topic_{topic1}']],
            y=[row[f'topic_{topic2}']],
            mode='markers+text',
            marker=dict(size=15, color=row['color']),
            text=row['movement'],
            textposition="top center",
            name=row['movement'],
            hovertext=f"{row['movement']}<br>{row['text'][:100]}..."
        ))
    
    fig.update_layout(
        title=f"Political Movements: {topic1.capitalize()} vs {topic2.capitalize()}",
        xaxis=dict(title=f"{topic1.capitalize()} Affinity", range=[0, 1]),
        yaxis=dict(title=f"{topic2.capitalize()} Affinity", range=[0, 1]),
        hovermode='closest'
    )
    
    return fig

# Create some interesting topic combinations
topic_pairs = [
    ('climate', 'economy'),
    ('migration', 'security'),
    ('social', 'europe')
]

for topic1, topic2 in topic_pairs:
    fig = create_topic_projection(topic1, topic2)
    fig.write_html(f"political_movement_{topic1}_vs_{topic2}.html")
    print(f"Saved {topic1} vs {topic2} projection to political_movement_{topic1}_vs_{topic2}.html")

print("\nDone! You can now explore the topic-based visualizations.")
