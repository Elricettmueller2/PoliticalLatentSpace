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
    
    # Get word cloud if embeddings are available
    word_cloud = []
    if EMBEDDING_STORE is not None:
        print(f"Attempting to get word cloud for {entity_type} {entity_name} from embeddings...")
        
        # Get entity embedding
        entity_embedding = LATENT_SPACE.get_entity_embedding(entity_type, entity_name)
        
        if entity_embedding is not None:
            print(f"Entity embedding shape: {entity_embedding.shape}")
            print(f"Entity embedding norm: {np.linalg.norm(entity_embedding)}")
            print(f"Entity embedding sample: {entity_embedding[:5]}")
            
            # Check a few sample words for debugging
            print("Checking word embeddings...")
            for sample_word in ['politik', 'demokratie', 'wirtschaft']:
                word_embedding = EMBEDDING_STORE.get_word_embedding(sample_word)
                if word_embedding is not None:
                    word_norm = np.linalg.norm(word_embedding)
                    entity_norm = np.linalg.norm(entity_embedding)
                    similarity = np.dot(word_embedding, entity_embedding) / (word_norm * entity_norm)
                    print(f"Word '{sample_word}' embedding norm: {word_norm}")
                    print(f"Similarity to '{entity_name}': {similarity}")
            
            # Get nearest words
            nearest_words = EMBEDDING_STORE.get_nearest_words(
                entity_embedding, 
                top_n=50,
                max_distance=5.0
            )
            
            if nearest_words:
                print(f"Found {len(nearest_words)} words for word cloud")
                print(f"First few words: {[w['word'] for w in nearest_words[:5]]}")
                
                # Format for word cloud
                word_cloud = [
                    {
                        'text': w['word'],
                        'value': 1.0 - (w['distance'] / 5.0)  # Convert distance to similarity
                    }
                    for w in nearest_words
                ]
                
                print(f"Generated word cloud with {len(word_cloud)} words from embeddings")
    
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
    if EMBEDDING_STORE is None:
        return jsonify({
            'error': 'Word embedding store not available',
            'message': 'Embeddings have not been loaded. Please check server configuration.'
        }), 503
    
    # Get query parameters
    entity_type = request.args.get('entity_type')
    entity_name = request.args.get('entity_name')
    top_n = int(request.args.get('top_n', 100))
    max_distance = float(request.args.get('max_distance', 5.0))  # Updated default to 5.0
    filter_stopwords = request.args.get('filter_stopwords', 'true').lower() == 'true'  # Add stopword filtering
    
    if not entity_type or not entity_name:
        return jsonify({
            'error': 'Missing parameters',
            'message': 'Both entity_type and entity_name must be provided'
        }), 400
    
    # Map entity name to lowercase for consistent lookup
    entity_name = entity_name.lower()
    
    # Get entity embedding
    entity_embedding = LATENT_SPACE.get_entity_embedding(entity_type, entity_name)
    if entity_embedding is None:
        return jsonify({
            'error': 'Entity not found',
            'message': f'No embedding found for {entity_type} {entity_name}'
        }), 404
    
    # Find nearest words
    nearest_words = LATENT_SPACE.get_word_cloud_for_entity(
        entity_type,
        entity_name,
        top_n=top_n,
        max_distance=max_distance,
        filter_stopwords=filter_stopwords
    )
    
    # Format for word cloud
    word_cloud_data = [
        {
            'text': w.get('text', w.get('word', '')),  # Handle both 'text' and 'word' field names
            'value': w.get('value', 1.0 - w.get('distance', 0)),  # Convert distance to similarity if needed
            'distance': w.get('distance', 0),
            'position': w.get('position', [0, 0, 0])  # Include position data for layout
        }
        for w in nearest_words
    ]
    
    return jsonify({
        'entity': {
            'type': entity_type,
            'name': entity_name
        },
        'word_cloud': word_cloud_data
    })

if __name__ == '__main__':
    app.run(debug=True)
