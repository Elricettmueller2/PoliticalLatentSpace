#!/usr/bin/env python3
"""
Interactive Latent Space Explorer

This script provides an interactive visualization tool for exploring
the relationships between words and entities in the political latent space.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import argparse

# Add the project root to the path so we can import the modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our analyzer
from scripts.analyze_latent_space import LatentSpaceAnalyzer

class LatentSpaceExplorer:
    """
    A class for interactively exploring the political latent space.
    """
    
    def __init__(self):
        """Initialize the LatentSpaceExplorer."""
        self.analyzer = LatentSpaceAnalyzer()
        self.latent_space = self.analyzer.latent_space
        
        # Get lists of entities
        self.movements = self.analyzer.movements
        self.politicians = self.analyzer.politicians
        
        # Political spectrum colors (from focus_view.js)
        self.spectrum_colors = {
            'far-left': '#FF0000',      # Red
            'left': '#FF6666',          # Light red
            'center-left': '#FFC0CB',   # Pink
            'center': '#AAAAAA',        # Gray
            'center-right': '#ADD8E6',  # Light blue
            'right': '#6666FF',         # Light blue
            'far-right': '#0000FF'      # Blue
        }
    
    def get_entity_embeddings(self, entity_types=None):
        """
        Get embeddings for all entities of the specified types.
        
        Args:
            entity_types: List of entity types to include (if None, include all)
            
        Returns:
            DataFrame with entity information and embeddings
        """
        if entity_types is None:
            entity_types = ['movement', 'politician']
        
        entities = []
        
        for entity_type in entity_types:
            entity_list = self.movements if entity_type == 'movement' else self.politicians
            
            for entity_name in entity_list:
                embedding = self.latent_space.get_entity_embedding(entity_type, entity_name)
                
                if embedding is not None:
                    # Get entity data
                    if entity_type == 'movement':
                        entity_data = self.analyzer.entity_data['movements'].get(entity_name, {})
                    else:
                        entity_data = self.analyzer.entity_data['politicians'].get(entity_name, {})
                    
                    # Get political spectrum
                    spectrum = entity_data.get('political_spectrum', 'center')
                    
                    # Add to entities list
                    entities.append({
                        'type': entity_type,
                        'name': entity_name,
                        'embedding': embedding,
                        'political_spectrum': spectrum,
                        'color': self.spectrum_colors.get(spectrum, '#AAAAAA')
                    })
        
        return pd.DataFrame(entities)
    
    def get_word_embeddings(self, words):
        """
        Get embeddings for the specified words.
        
        Args:
            words: List of words to get embeddings for
            
        Returns:
            DataFrame with word information and embeddings
        """
        word_data = []
        
        for word in words:
            embedding = self.latent_space.embedding_store.get_word_embedding(word)
            
            if embedding is not None:
                word_data.append({
                    'type': 'word',
                    'name': word,
                    'embedding': embedding,
                    'color': '#00AA00'  # Green for words
                })
        
        return pd.DataFrame(word_data)
    
    def get_nearest_words_to_entity(self, entity_type, entity_name, top_n=50):
        """
        Get the nearest words to an entity.
        
        Args:
            entity_type: Type of entity ('movement' or 'politician')
            entity_name: Name of the entity
            top_n: Number of words to include
            
        Returns:
            DataFrame with word information and embeddings
        """
        word_cloud = self.analyzer.get_word_cloud(entity_type, entity_name, top_n=top_n)
        words = [item['text'] for item in word_cloud]
        
        return self.get_word_embeddings(words)
    
    def create_interactive_visualization(self, port=8050):
        """
        Create an interactive visualization of the latent space.
        
        Args:
            port: Port to run the Dash app on
            
        Returns:
            Dash app
        """
        # Get entity embeddings
        entity_df = self.get_entity_embeddings()
        
        # Initialize the Dash app
        app = dash.Dash(__name__)
        
        # Define the layout
        app.layout = html.Div([
            html.H1("Political Latent Space Explorer"),
            
            html.Div([
                html.Div([
                    html.H3("Entity Selection"),
                    html.Label("Entity Type:"),
                    dcc.Dropdown(
                        id='entity-type-dropdown',
                        options=[
                            {'label': 'Movement', 'value': 'movement'},
                            {'label': 'Politician', 'value': 'politician'}
                        ],
                        value='movement'
                    ),
                    html.Label("Entity Name:"),
                    dcc.Dropdown(id='entity-name-dropdown'),
                    html.Button('Add Entity', id='add-entity-button', n_clicks=0),
                    
                    html.H3("Word Selection"),
                    html.Label("Word:"),
                    dcc.Input(id='word-input', type='text', value=''),
                    html.Button('Add Word', id='add-word-button', n_clicks=0),
                    html.Button('Add Top Words for Entity', id='add-top-words-button', n_clicks=0),
                    
                    html.H3("Visualization Options"),
                    html.Label("Number of Top Words:"),
                    dcc.Slider(
                        id='top-n-slider',
                        min=10,
                        max=100,
                        step=10,
                        value=50,
                        marks={i: str(i) for i in range(10, 101, 10)}
                    ),
                    html.Label("Projection Method:"),
                    dcc.RadioItems(
                        id='projection-method',
                        options=[
                            {'label': 'PCA', 'value': 'pca'},
                            {'label': 't-SNE', 'value': 'tsne'}
                        ],
                        value='tsne'
                    ),
                    html.Button('Reset Visualization', id='reset-button', n_clicks=0),
                ], style={'width': '30%', 'float': 'left', 'padding': '20px'}),
                
                html.Div([
                    dcc.Graph(id='latent-space-graph', style={'height': '80vh'})
                ], style={'width': '70%', 'float': 'right'})
            ], style={'display': 'flex', 'flexDirection': 'row'}),
            
            # Store components for the current state
            dcc.Store(id='selected-entities', data=[]),
            dcc.Store(id='selected-words', data=[])
        ])
        
        # Update entity name dropdown based on entity type
        @app.callback(
            Output('entity-name-dropdown', 'options'),
            Input('entity-type-dropdown', 'value')
        )
        def update_entity_names(entity_type):
            if entity_type == 'movement':
                return [{'label': name, 'value': name} for name in self.movements]
            else:
                return [{'label': name, 'value': name} for name in self.politicians]
        
        # Set default value for entity name dropdown
        @app.callback(
            Output('entity-name-dropdown', 'value'),
            Input('entity-name-dropdown', 'options')
        )
        def set_default_entity(options):
            if options:
                return options[0]['value']
            return None
        
        # Add entity to the visualization
        @app.callback(
            Output('selected-entities', 'data'),
            Input('add-entity-button', 'n_clicks'),
            State('entity-type-dropdown', 'value'),
            State('entity-name-dropdown', 'value'),
            State('selected-entities', 'data')
        )
        def add_entity(n_clicks, entity_type, entity_name, selected_entities):
            if n_clicks == 0:
                return selected_entities
            
            # Check if entity is already selected
            for entity in selected_entities:
                if entity['type'] == entity_type and entity['name'] == entity_name:
                    return selected_entities
            
            # Get entity data
            embedding = self.latent_space.get_entity_embedding(entity_type, entity_name)
            
            if embedding is not None:
                # Get entity data
                if entity_type == 'movement':
                    entity_data = self.analyzer.entity_data['movements'].get(entity_name, {})
                else:
                    entity_data = self.analyzer.entity_data['politicians'].get(entity_name, {})
                
                # Get political spectrum
                spectrum = entity_data.get('political_spectrum', 'center')
                
                # Add to selected entities
                selected_entities.append({
                    'type': entity_type,
                    'name': entity_name,
                    'political_spectrum': spectrum,
                    'color': self.spectrum_colors.get(spectrum, '#AAAAAA')
                })
            
            return selected_entities
        
        # Add word to the visualization
        @app.callback(
            Output('selected-words', 'data'),
            Input('add-word-button', 'n_clicks'),
            Input('add-top-words-button', 'n_clicks'),
            State('word-input', 'value'),
            State('entity-type-dropdown', 'value'),
            State('entity-name-dropdown', 'value'),
            State('top-n-slider', 'value'),
            State('selected-words', 'data')
        )
        def add_word(add_word_clicks, add_top_clicks, word, entity_type, entity_name, top_n, selected_words):
            ctx = dash.callback_context
            if not ctx.triggered:
                return selected_words
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if trigger_id == 'add-word-button' and word:
                # Check if word is already selected
                if word in [w['name'] for w in selected_words]:
                    return selected_words
                
                # Get word embedding
                embedding = self.latent_space.embedding_store.get_word_embedding(word)
                
                if embedding is not None:
                    # Add to selected words
                    selected_words.append({
                        'name': word,
                        'color': '#00AA00'  # Green for words
                    })
            
            elif trigger_id == 'add-top-words-button':
                # Get top words for entity
                word_cloud = self.analyzer.get_word_cloud(entity_type, entity_name, top_n=top_n)
                
                # Add words to selected words
                for item in word_cloud:
                    word = item['text']
                    
                    # Check if word is already selected
                    if word not in [w['name'] for w in selected_words]:
                        selected_words.append({
                            'name': word,
                            'color': '#00AA00'  # Green for words
                        })
            
            return selected_words
        
        # Reset visualization
        @app.callback(
            [Output('selected-entities', 'data', allow_duplicate=True),
             Output('selected-words', 'data', allow_duplicate=True)],
            Input('reset-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def reset_visualization(n_clicks):
            return [], []
        
        # Update the visualization
        @app.callback(
            Output('latent-space-graph', 'figure'),
            [Input('selected-entities', 'data'),
             Input('selected-words', 'data'),
             Input('projection-method', 'value')]
        )
        def update_visualization(selected_entities, selected_words, projection_method):
            # Prepare data for visualization
            data = []
            embeddings = []
            
            # Add entities
            for entity in selected_entities:
                entity_type = entity['type']
                entity_name = entity['name']
                
                embedding = self.latent_space.get_entity_embedding(entity_type, entity_name)
                
                if embedding is not None:
                    data.append({
                        'type': entity_type,
                        'name': entity_name,
                        'color': entity['color']
                    })
                    embeddings.append(embedding)
            
            # Add words
            for word in selected_words:
                word_name = word['name']
                
                embedding = self.latent_space.embedding_store.get_word_embedding(word_name)
                
                if embedding is not None:
                    data.append({
                        'type': 'word',
                        'name': word_name,
                        'color': word['color']
                    })
                    embeddings.append(embedding)
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # If no data, return empty figure
            if len(df) == 0:
                return go.Figure()
            
            # Convert embeddings to numpy array
            embeddings = np.array(embeddings)
            
            # Project embeddings to 2D
            if projection_method == 'pca':
                from sklearn.decomposition import PCA
                projection = PCA(n_components=2).fit_transform(embeddings)
            else:  # t-SNE
                projection = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
            
            # Add projection to DataFrame
            df['x'] = projection[:, 0]
            df['y'] = projection[:, 1]
            
            # Create figure
            fig = px.scatter(
                df, x='x', y='y', color='type',
                hover_name='name', text='name',
                color_discrete_map={
                    'movement': '#0000FF',
                    'politician': '#FF0000',
                    'word': '#00AA00'
                }
            )
            
            # Update traces
            for i, row in df.iterrows():
                fig.add_annotation(
                    x=row['x'],
                    y=row['y'],
                    text=row['name'],
                    showarrow=False,
                    font=dict(
                        size=10,
                        color="black"
                    ),
                    bgcolor="white",
                    opacity=0.8
                )
            
            # Update layout
            fig.update_layout(
                title='Political Latent Space',
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                legend_title='Entity Type',
                height=800
            )
            
            return fig
        
        return app
    
    def run_interactive_visualization(self, port=8050):
        """
        Run the interactive visualization.
        
        Args:
            port: Port to run the Dash app on
        """
        app = self.create_interactive_visualization()
        app.run_server(debug=True, port=port)

def main():
    """Main function to run the latent space explorer."""
    parser = argparse.ArgumentParser(description='Explore the political latent space')
    
    parser.add_argument('--port', type=int, default=8050,
                        help='Port to run the Dash app on')
    
    args = parser.parse_args()
    
    explorer = LatentSpaceExplorer()
    explorer.run_interactive_visualization(port=args.port)

if __name__ == "__main__":
    main()
