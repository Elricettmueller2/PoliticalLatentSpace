from flask import Flask, render_template, jsonify, request
import json
import plotly
import random
import os
import numpy as np
from pathlib import Path

# Import visualization and latent space functions
from prototypes.galaxy_visualization import create_galaxy_visualization
from src.data.embeddings.multi_level_latent_space import MultiLevelLatentSpace
from src.data.embeddings.chunked_embedding_store import ChunkedEmbeddingStore

# --- This is the new import statement that brings in the correct logic ---
from scripts.analyze_latent_space import LatentSpaceAnalyzer

app = Flask(__name__)

# --- Data Loading and Pre-computation ---
print("Loading and processing data...")

# Direct path to the multi_level_analysis_results.json file
ANALYSIS_RESULTS_PATH = 'src/data/processed/multi_level_analysis_results.json'
WORD_EMBEDDING_PATH = 'src/data/processed/word_embeddings.h5'
WORD_INDEX_PATH = 'src/data/processed/word_embeddings.index'

# Load the data directly from the JSON file
DATA = {}
try:
    print(f"Loading data directly from {ANALYSIS_RESULTS_PATH}...")
    with open(ANALYSIS_RESULTS_PATH, 'r', encoding='utf-8') as f:
        DATA = json.load(f)
    print(f"Data loaded successfully with keys: {list(DATA.keys())}")
except Exception as e:
    print(f"Error loading data: {e}")
    # Initialize with empty data if file not found
    DATA = {'movements': {}, 'politicians': {}}

# Count entities
print(f"Found {len(DATA.get('movements', {}))} movements")
print(f"Found {len(DATA.get('politicians', {}))} politicians")

# Load word embeddings
EMBEDDING_STORE = None
try:
    print(f"Loading word embeddings from {WORD_EMBEDDING_PATH}...")
    EMBEDDING_STORE = ChunkedEmbeddingStore(WORD_EMBEDDING_PATH, WORD_INDEX_PATH)
    print("Word embeddings loaded successfully")
except Exception as e:
    print(f"Error loading word embeddings: {e}")

# Initialize latent space
print("Initializing latent space...")
LATENT_SPACE = MultiLevelLatentSpace(
    entity_data_path=ANALYSIS_RESULTS_PATH,
    word_embedding_path=WORD_EMBEDDING_PATH,
    index_file=WORD_INDEX_PATH,
    verbose=True
)
print("Latent space initialized successfully")

# Check if entity embeddings are available
print("Checking entity embeddings...")
sample_movement = list(DATA.get('movements', {}).keys())[0] if DATA.get('movements') else None
sample_politician = list(DATA.get('politicians', {}).keys())[0] if DATA.get('politicians') else None

if sample_movement:
    has_movement_embedding = LATENT_SPACE.get_entity_embedding('movement', sample_movement) is not None
    print(f"Sample movement '{sample_movement}' has embedding: {has_movement_embedding}")

if sample_politician:
    has_politician_embedding = LATENT_SPACE.get_entity_embedding('politician', sample_politician) is not None
    print(f"Sample politician '{sample_politician}' has embedding: {has_politician_embedding}")

print("Data loading complete.")

# --- Initialize the Latent Space Analyzer ---
print("Initializing Latent Space Analyzer...")
ANALYZER = LatentSpaceAnalyzer(
    entity_data_path=ANALYSIS_RESULTS_PATH,
    word_embedding_path=WORD_EMBEDDING_PATH,
    index_file=WORD_INDEX_PATH
)
print("Latent Space Analyzer initialized successfully.")

# --- Routes ---

@app.route('/')
def index():
    """Render the main visualization page."""
    return render_template('index.html')

@app.route('/api/galaxy-data')
def get_galaxy_data():
    """
    Generate and return the galaxy visualization data.
    
    Returns:
        JSON with visualization data
    """
    print("Data keys:", list(DATA.keys()))
    print(f"Found {len(DATA.get('movements', {}))} movements")
    print(f"Found {len(DATA.get('politicians', {}))} politicians")
    
    # Create the visualization
    figure = create_galaxy_visualization(DATA)
    
    # Convert Plotly Figure to JSON using plotly's serialization
    visualization_data = json.loads(plotly.io.to_json(figure))
    
    # Return the JSON data
    return jsonify(visualization_data)

@app.route('/api/entity-focus')
def get_entity_focus():
    """
    Get detailed information about a specific entity for the focus view.
    
    Query parameters:
    - entity_type: Type of entity (movement, politician)
    - entity_name: Name of entity
    
    Returns:
        JSON with entity details and related entities
    """
    # Get query parameters
    entity_type = request.args.get('entity_type')
    entity_name = request.args.get('entity_name')
    
    if not entity_type or not entity_name:
        return jsonify({
            'error': 'Missing parameters',
            'message': 'Both entity_type and entity_name must be provided'
        }), 400
    
    # Map entity name to lowercase for consistent lookup
    entity_name = entity_name.lower()
    
    print(f"\n--- Entity Focus Request: {entity_type} {entity_name} ---")
    
    # Get entity data
    entity_data = None
    if entity_type == 'movement':
        entity_data = DATA.get('movements', {}).get(entity_name)
    elif entity_type == 'politician':
        entity_data = DATA.get('politicians', {}).get(entity_name)
    
    if not entity_data:
        return jsonify({
            'error': 'Entity not found',
            'message': f'No data found for {entity_type} {entity_name}'
        }), 404
    
    # Get related entities
    related_entities = []
    
    if entity_type == 'movement':
        # For a movement, find all politicians belonging to it
        for politician_name, politician_data in DATA.get('politicians', {}).items():
            if politician_data.get('movement', '').lower() == entity_name:
                related_entities.append({
                    'type': 'politician',
                    'name': politician_name,
                    'relation': 'member'
                })
    elif entity_type == 'politician':
        # For a politician, add their movement
        movement = entity_data.get('movement', '')
        if movement:
            related_entities.append({
                'type': 'movement',
                'name': movement,
                'relation': 'belongs_to'
            })
        
        # Find other politicians from the same movement
        movement_lower = movement.lower()
        for politician_name, politician_data in DATA.get('politicians', {}).items():
            if (politician_name.lower() != entity_name.lower() and 
                politician_data.get('movement', '').lower() == movement_lower):
                related_entities.append({
                    'type': 'politician',
                    'name': politician_name,
                    'relation': 'colleague'
                })
    
    # --- Word Cloud Generation (for initial focus view) ---
    # Use the analyzer to get a consistent word cloud
    word_cloud = ANALYZER.get_word_cloud(
        entity_type=entity_type,
        entity_name=entity_name,
        top_n=50,  # A reasonable default for the initial view
        max_distance=5.0,
        filter_stopwords=True
    )
    if word_cloud is None:
        word_cloud = [] # Default to empty list if no cloud can be generated

    # Prepare response
    response = {
        'entity': {
            'type': entity_type,
            'name': entity_name,
            'data': entity_data
        },
        'related_entities': related_entities,
        'word_cloud': word_cloud
    }
    
    return jsonify(response)

@app.route('/api/word-cloud/entity')
def get_entity_word_cloud():
    """
    Generate a word cloud for a specific entity based on nearby words in the embedding space.
    
    Query parameters:
    - entity_type: Type of entity (party, politician)
    - entity_name: Name of entity
    - top_n: Number of words to include (default: 100)
    - max_distance: Maximum distance threshold (default: 5.0)
    - filter_stopwords: Whether to filter out common German stopwords (default: true)
    """
    # Get query parameters
    entity_type = request.args.get('entity_type')
    entity_name = request.args.get('entity_name')
    top_n = int(request.args.get('top_n', 100))
    max_distance = float(request.args.get('max_distance', 5.0))
    filter_stopwords = request.args.get('filter_stopwords', 'true').lower() == 'true'
    
    if not entity_type or not entity_name:
        return jsonify({
            'error': 'Missing parameters',
            'message': 'Both entity_type and entity_name must be provided'
        }), 400
    
    # Use the analyzer to get the word cloud, which handles stopword filtering correctly
    word_cloud_data = ANALYZER.get_word_cloud(
        entity_type=entity_type,
        entity_name=entity_name,
        top_n=top_n,
        max_distance=max_distance,
        filter_stopwords=filter_stopwords
    )
    
    if word_cloud_data is None:
        return jsonify({
            'error': 'Entity not found',
            'message': f'Could not generate word cloud for {entity_type} {entity_name}'
        }), 404

    return jsonify({
        'entity': {
            'type': entity_type,
            'name': entity_name
        },
        'word_cloud': word_cloud_data
    })

if __name__ == '__main__':
    app.run(debug=True)
