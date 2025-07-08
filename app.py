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

app = Flask(__name__)

# --- Data Loading and Pre-computation ---
print("Loading and processing data...")

# Direct path to the multi_level_analysis_results.json file
ANALYSIS_RESULTS_PATH = 'src/data/processed/multi_level_analysis_results.json'

# Load the data directly from the JSON file
DATA = {}
try:
    print(f"Loading data directly from {ANALYSIS_RESULTS_PATH}...")
    with open(ANALYSIS_RESULTS_PATH, 'r', encoding='utf-8') as f:
        DATA = json.load(f)
    print(f"Data loaded successfully with keys: {list(DATA.keys())}")
    if 'movements' in DATA:
        print(f"Found {len(DATA['movements'])} movements")
    if 'politicians' in DATA:
        print(f"Found {len(DATA['politicians'])} politicians")
    
    # Initialize empty embeddings dictionary if needed
    if 'embeddings' not in DATA:
        DATA['embeddings'] = {}
        
except Exception as e:
    print(f"Error loading data: {e}")
    DATA = {'movements': {}, 'politicians': {}, 'embeddings': {}}

# No word embeddings in this simplified version
EMBEDDING_STORE = None
LATENT_SPACE = None

print("Data loading complete.")


# --- Routes ---

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/api/galaxy-data')
def get_galaxy_data():
    """Provide the data for the galaxy plot in a JSON format compatible with Plotly.js."""
    # Get selected entity from query parameters
    entity_type = request.args.get('entity_type')
    entity_name = request.args.get('entity_name')
    
    selected_entity = None
    if entity_type and entity_name:
        selected_entity = {'type': entity_type, 'name': entity_name}
    
    try:
        # Simply use the create_galaxy_visualization function with our loaded data
        galaxy_fig = create_galaxy_visualization(DATA, selected_entity=selected_entity)
        
        # Use Plotly's JSON encoder to serialize the figure
        return json.dumps(galaxy_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
    except Exception as e:
        print(f"Error generating galaxy visualization: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a valid JSON error response instead of throwing a 500
        error_response = {
            'error': True,
            'message': f"Failed to generate visualization: {str(e)}",
            'data': [],
            'layout': {
                'title': 'Error: Could not load visualization',
                'annotations': [{
                    'text': f"Error: {str(e)}",
                    'showarrow': False,
                    'font': {'size': 16, 'color': 'red'},
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5
                }]
            }
        }
        return jsonify(error_response)

# Dictionary of political terms for generating sample word clouds
POLITICAL_TERMS = {
    'economic': ['market', 'economy', 'taxes', 'budget', 'inflation', 'jobs', 'growth', 'debt', 'deficit', 'trade', 'investment', 'regulation', 'subsidies', 'privatization', 'austerity'],
    'social': ['equality', 'justice', 'rights', 'freedom', 'liberty', 'community', 'welfare', 'education', 'healthcare', 'immigration', 'diversity', 'inclusion', 'tradition', 'family', 'security'],
    'ecological': ['climate', 'environment', 'sustainability', 'renewable', 'conservation', 'pollution', 'biodiversity', 'emissions', 'green', 'energy', 'resources', 'recycling', 'agriculture', 'wildlife', 'protection']
}

# Movement-specific terms
MOVEMENT_TERMS = {
    'cdu': ['conservative', 'christian', 'tradition', 'stability', 'market', 'security', 'europe', 'nato', 'family', 'business'],
    'spd': ['social', 'workers', 'solidarity', 'equality', 'welfare', 'labor', 'justice', 'reform', 'progress', 'rights'],
    'fdp': ['liberal', 'freedom', 'market', 'individual', 'business', 'deregulation', 'competition', 'innovation', 'taxes', 'privacy'],
    'gruene': ['environment', 'climate', 'sustainability', 'renewable', 'diversity', 'equality', 'peace', 'europe', 'future', 'protection'],
    'linke': ['socialism', 'equality', 'redistribution', 'workers', 'anti-capitalism', 'peace', 'solidarity', 'public', 'welfare', 'justice'],
    'afd': ['immigration', 'tradition', 'sovereignty', 'identity', 'security', 'family', 'nation', 'euro-skeptic', 'borders', 'culture'],
    'volt': ['europe', 'integration', 'progressive', 'digital', 'innovation', 'climate', 'transparency', 'mobility', 'education', 'cooperation'],
    'bsw': ['peace', 'diplomacy', 'social', 'justice', 'sovereignty', 'reform', 'dialogue', 'stability', 'pragmatic', 'security']
}

# Political dimension scores for words to position them in the latent space
# Format: {word: [economic_score, social_score, ecological_score]}
# Scores range from -1.0 to 1.0 where:
# - Economic: Market Liberal (+) vs. Social Democratic (-)
# - Social: Progressive (+) vs. Conservative (-)
# - Ecological: Green (+) vs. Industrial (-)
WORD_DIMENSION_SCORES = {
    # Economic terms
    'market': [0.9, 0.0, -0.2],        # Strongly market liberal
    'economy': [0.7, 0.0, -0.1],       # Market liberal
    'taxes': [0.5, 0.0, 0.0],          # Somewhat market liberal
    'budget': [0.6, 0.0, 0.0],         # Market liberal
    'inflation': [0.4, -0.1, 0.0],     # Somewhat market liberal
    'jobs': [0.2, 0.3, 0.0],           # Slightly market liberal, slightly progressive
    'growth': [0.7, 0.2, -0.3],        # Market liberal, slightly progressive, somewhat industrial
    'debt': [0.3, -0.1, 0.0],          # Somewhat market liberal
    'deficit': [0.4, -0.1, 0.0],       # Somewhat market liberal
    'trade': [0.8, 0.1, -0.1],         # Strongly market liberal
    'investment': [0.6, 0.2, 0.0],     # Market liberal, slightly progressive
    'regulation': [-0.5, 0.3, 0.5],    # Social democratic, progressive, somewhat green
    'subsidies': [-0.7, 0.1, 0.3],     # Strongly social democratic
    'privatization': [0.9, 0.0, -0.2], # Strongly market liberal
    'austerity': [0.8, -0.3, -0.1],    # Strongly market liberal, somewhat conservative
    'business': [0.8, 0.0, -0.3],      # Strongly market liberal, somewhat industrial
    'deregulation': [0.9, -0.1, -0.5], # Strongly market liberal, industrial
    'competition': [0.8, 0.2, -0.1],   # Strongly market liberal
    'innovation': [0.5, 0.7, 0.3],     # Market liberal, progressive
    
    # Social terms
    'equality': [-0.5, 0.8, 0.2],      # Social democratic, strongly progressive
    'justice': [-0.3, 0.6, 0.1],       # Somewhat social democratic, progressive
    'rights': [0.1, 0.8, 0.1],         # Progressive
    'freedom': [0.5, 0.7, 0.0],        # Market liberal, progressive
    'liberty': [0.6, 0.6, 0.0],        # Market liberal, progressive
    'community': [-0.3, 0.5, 0.3],     # Somewhat social democratic, progressive
    'welfare': [-0.8, 0.5, 0.0],       # Strongly social democratic, progressive
    'education': [-0.2, 0.7, 0.2],     # Slightly social democratic, progressive
    'healthcare': [-0.7, 0.6, 0.1],    # Strongly social democratic, progressive
    'immigration': [-0.1, 0.8, 0.1],   # Strongly progressive
    'diversity': [-0.2, 0.9, 0.2],     # Strongly progressive
    'inclusion': [-0.3, 0.9, 0.2],     # Strongly progressive
    'tradition': [0.1, -0.8, -0.1],    # Strongly conservative
    'family': [0.0, -0.5, 0.0],        # Conservative
    'security': [0.2, -0.4, -0.1],     # Conservative
    'workers': [-0.8, 0.4, 0.0],       # Strongly social democratic, somewhat progressive
    'solidarity': [-0.7, 0.5, 0.1],    # Strongly social democratic, progressive
    'reform': [0.0, 0.6, 0.2],         # Progressive
    'progress': [0.1, 0.8, 0.3],       # Strongly progressive
    'individual': [0.7, 0.5, 0.0],     # Market liberal, progressive
    'privacy': [0.3, 0.7, 0.0],        # Progressive
    'identity': [0.0, -0.5, 0.0],      # Conservative
    'sovereignty': [0.2, -0.6, -0.1],  # Conservative
    'nation': [0.1, -0.7, -0.2],       # Strongly conservative
    'borders': [0.2, -0.8, -0.1],      # Strongly conservative
    'culture': [0.0, -0.4, 0.1],       # Somewhat conservative
    
    # Ecological terms
    'climate': [-0.2, 0.5, 0.9],       # Social democratic, progressive, strongly green
    'environment': [-0.3, 0.4, 0.9],   # Somewhat social democratic, progressive, strongly green
    'sustainability': [-0.2, 0.5, 0.9], # Social democratic, progressive, strongly green
    'renewable': [-0.3, 0.4, 0.9],     # Somewhat social democratic, progressive, strongly green
    'conservation': [-0.1, 0.2, 0.8],   # Green
    'pollution': [-0.2, 0.3, 0.8],      # Social democratic, somewhat progressive, strongly green
    'biodiversity': [-0.2, 0.4, 0.9],   # Social democratic, progressive, strongly green
    'emissions': [-0.2, 0.3, 0.8],      # Social democratic, somewhat progressive, strongly green
    'green': [-0.3, 0.5, 0.9],         # Somewhat social democratic, progressive, strongly green
    'energy': [0.0, 0.2, 0.6],         # Somewhat green
    'resources': [0.0, 0.0, 0.5],      # Somewhat green
    'recycling': [-0.2, 0.3, 0.8],     # Social democratic, somewhat progressive, strongly green
    'agriculture': [0.0, -0.1, 0.5],    # Somewhat green
    'wildlife': [-0.1, 0.3, 0.9],      # Strongly green
    'protection': [-0.3, 0.2, 0.7],    # Somewhat social democratic, green
    
    # Movement-specific terms
    'christian': [0.2, -0.7, 0.0],      # Conservative
    'stability': [0.3, -0.3, 0.0],      # Somewhat market liberal, somewhat conservative
    'nato': [0.3, 0.0, -0.2],          # Somewhat market liberal
    'socialism': [-0.9, 0.3, 0.2],     # Strongly social democratic
    'redistribution': [-0.9, 0.4, 0.1], # Strongly social democratic
    'anti-capitalism': [-0.9, 0.2, 0.3], # Strongly social democratic
    'public': [-0.7, 0.3, 0.1],        # Strongly social democratic
    'liberal': [0.7, 0.6, 0.0],        # Market liberal, progressive
    'euro-skeptic': [0.1, -0.7, -0.1], # Strongly conservative
    'europe': [0.2, 0.5, 0.3],         # Somewhat market liberal, progressive
    'integration': [0.0, 0.7, 0.2],    # Progressive
    'progressive': [0.0, 0.9, 0.3],    # Strongly progressive
    'digital': [0.4, 0.7, 0.1],        # Somewhat market liberal, progressive
    'transparency': [0.1, 0.8, 0.2],   # Strongly progressive
    'mobility': [0.2, 0.6, 0.5],       # Somewhat market liberal, progressive, somewhat green
    'peace': [-0.2, 0.5, 0.3],         # Somewhat social democratic, progressive
    'diplomacy': [-0.1, 0.4, 0.2],     # Progressive
    'dialogue': [0.0, 0.6, 0.1],       # Progressive
    'pragmatic': [0.3, 0.1, 0.0],      # Somewhat market liberal
    'future': [0.1, 0.6, 0.5]          # Progressive, somewhat green
}

def generate_word_cloud(entity_type, entity_name):
    """Generate a sample word cloud for an entity when none exists in the data."""
    word_cloud = {}
    
    # Base terms from all categories with random weights
    all_terms = POLITICAL_TERMS['economic'] + POLITICAL_TERMS['social'] + POLITICAL_TERMS['ecological']
    for term in random.sample(all_terms, 15):  # Select 15 random terms
        word_cloud[term] = round(random.uniform(0.1, 1.0), 2)
    
    # Add movement-specific terms if it's a movement or a politician from that movement
    if entity_type == 'movement' and entity_name.lower() in MOVEMENT_TERMS:
        for term in MOVEMENT_TERMS[entity_name.lower()]:  # Add all movement-specific terms
            word_cloud[term] = round(random.uniform(0.5, 1.0), 2)  # Higher weights for movement terms
    elif entity_type == 'politician':
        # Try to find the politician's movement
        politician_data = DATA['politicians'].get(entity_name, {})
        movement = politician_data.get('movement', '').lower()
        if movement in MOVEMENT_TERMS:
            for term in random.sample(MOVEMENT_TERMS[movement], 5):  # Add 5 random terms from their movement
                word_cloud[term] = round(random.uniform(0.5, 1.0), 2)
    
    return word_cloud

@app.route('/api/entity-info')
def get_entity_info():
    """Provide detailed information about a specific entity including word clouds."""
    entity_type = request.args.get('entity_type')
    entity_name = request.args.get('entity_name')
    
    if not entity_type or not entity_name:
        return jsonify({'error': 'Missing entity_type or entity_name parameter'}), 400
    
    # Get entity data
    entity_data = None
    if entity_type == 'movement' and entity_name in DATA['movements']:
        entity_data = DATA['movements'][entity_name]
    elif entity_type == 'politician' and entity_name in DATA['politicians']:
        entity_data = DATA['politicians'][entity_name]
    
    if not entity_data:
        return jsonify({'error': 'Entity not found'}), 404
    
    # Extract dimension scores
    dimension_scores = entity_data.get('position', {}).get('expert_dimensions', {}).get('axes', {})
    
    # Get word cloud data or generate if not present
    word_cloud = entity_data.get('word_cloud', {})
    if not word_cloud:  # If word cloud is empty, generate sample data
        word_cloud = generate_word_cloud(entity_type, entity_name)
    
    # Prepare result
    result = {
        'name': entity_name,
        'type': entity_type,
        'dimension_scores': dimension_scores,
        'word_cloud': word_cloud
    }
    
    return jsonify(result)

@app.route('/api/word-embeddings/nearest')
def get_nearest_words():
    """
    Find words closest to a given entity or embedding vector.
    
    Query parameters:
    - entity_type: Type of entity (party, politician)
    - entity_name: Name of entity
    - embedding: JSON array of embedding vector (alternative to entity)
    - top_n: Number of words to return (default: 50)
    - max_distance: Maximum distance threshold (default: 0.5)
    """
    if EMBEDDING_STORE is None:
        return jsonify({
            'error': 'Word embedding store not available',
            'message': 'Embeddings have not been loaded. Please check server configuration.'
        }), 503
    
    # Get query parameters
    entity_type = request.args.get('entity_type')
    entity_name = request.args.get('entity_name')
    embedding_json = request.args.get('embedding')
    top_n = int(request.args.get('top_n', 50))
    max_distance = float(request.args.get('max_distance', 0.5))
    
    # Get embedding vector
    query_embedding = None
    
    if embedding_json:
        # Use provided embedding vector
        try:
            query_embedding = json.loads(embedding_json)
        except json.JSONDecodeError:
            return jsonify({
                'error': 'Invalid embedding format',
                'message': 'Embedding must be a valid JSON array'
            }), 400
    elif entity_type and entity_name:
        # Get embedding for entity
        entity_key = f"{entity_type}:{entity_name}"
        if entity_key in DATA['embeddings']:
            query_embedding = DATA['embeddings'][entity_key]
        else:
            return jsonify({
                'error': 'Entity not found',
                'message': f'No embedding found for {entity_type} {entity_name}'
            }), 404
    else:
        return jsonify({
            'error': 'Missing parameters',
            'message': 'Either embedding or entity_type+entity_name must be provided'
        }), 400
    
    # Find nearest words
    nearest_words = EMBEDDING_STORE.get_nearest_words(
        query_embedding, 
        top_n=top_n,
        max_distance=max_distance
    )
    
    return jsonify({
        'query': {
            'entity_type': entity_type,
            'entity_name': entity_name,
            'top_n': top_n,
            'max_distance': max_distance
        },
        'nearest_words': nearest_words
    })

@app.route('/api/hybrid-visualization')
def get_hybrid_visualization():
    """
    Generate a hybrid visualization combining entity-level and word-level data.
    
    Query parameters:
    - entity_type: Type of entity to focus on (movement, politician)
    - entity_name: Name of entity to focus on
    - num_words: Number of words to include (default: 100)
    - max_distance: Maximum distance for word inclusion (default: 0.5)
    """
    if LATENT_SPACE is None or LATENT_SPACE.word_embedding_store is None:
        return jsonify({
            'error': 'Hybrid latent space not available',
            'message': 'The hybrid latent space has not been properly initialized. Please check server configuration.'
        }), 503
    
    # Get query parameters
    entity_type = request.args.get('entity_type', 'movement')
    entity_name = request.args.get('entity_name')
    num_words = int(request.args.get('num_words', 100))
    max_distance = float(request.args.get('max_distance', 0.5))
    
    # Create base visualization with entities
    selected_entity = None
    if entity_type and entity_name:
        selected_entity = {'type': entity_type, 'name': entity_name}
    
    # Generate the hybrid visualization directly from the latent space
    try:
        # Get the hybrid visualization data
        hybrid_data = LATENT_SPACE.get_hybrid_visualization_data(
            entity_type=entity_type,
            entity_name=entity_name,
            num_words=num_words,
            max_distance=max_distance
        )
        
        # If the latent space doesn't have the hybrid visualization method,
        # fall back to the old approach
        if hybrid_data is None:
            # Generate the entity visualization
            galaxy_fig = create_galaxy_visualization(DATA, selected_entity=selected_entity)
            
            # If entity is specified, add word-level data
            if entity_type and entity_name:
                entity_key = f"{entity_type}:{entity_name}"
                if 'embeddings' in DATA and entity_key in DATA['embeddings']:
                    # Get entity embedding
                    query_embedding = DATA['embeddings'][entity_key]
                    
                    # Find nearest words
                    nearest_words = EMBEDDING_STORE.get_nearest_words(
                        query_embedding, 
                        top_n=num_words,
                        max_distance=max_distance
                    )
                    
                    # Extract word data for visualization
                    words = [w['word'] for w in nearest_words]
                    distances = [w['distance'] for w in nearest_words]
                    
                    # Add word trace to visualization
                    if 'data' in galaxy_fig:
                        word_trace = {
                            'x': [w['position'][0] if 'position' in w and isinstance(w['position'], list) else 0 for w in nearest_words],
                            'y': [w['position'][1] if 'position' in w and isinstance(w['position'], list) and len(w['position']) > 1 else 0 for w in nearest_words],
                            'z': [w['position'][2] if 'position' in w and isinstance(w['position'], list) and len(w['position']) > 2 else 0 for w in nearest_words],
                            'text': words,
                            'mode': 'markers+text',
                            'marker': {
                                'size': [10 * (1 - d) for d in distances],  # Size inversely proportional to distance
                                'color': 'rgba(100, 100, 255, 0.7)',
                                'line': {'width': 1, 'color': 'white'}
                            },
                            'textfont': {
                                'size': 8,
                                'color': 'rgba(100, 100, 255, 1)'
                            },
                            'name': 'Related Words',
                            'hoverinfo': 'text',
                            'hovertext': [f"{w}: {d:.3f}" for w, d in zip(words, distances)]
                        }
                        galaxy_fig['data'].append(word_trace)
            
            # Convert to JSON
            return json.dumps(galaxy_fig, cls=plotly.utils.PlotlyJSONEncoder)
        else:
            # Convert the hybrid data to JSON
            return json.dumps(hybrid_data, cls=plotly.utils.PlotlyJSONEncoder)
            
    except Exception as e:
        print(f"Error generating hybrid visualization: {e}")
        return jsonify({
            'error': 'Failed to generate hybrid visualization',
            'message': str(e)
        }), 500

@app.route('/api/word-cloud/entity')
def get_entity_word_cloud():
    """
    Generate a word cloud for a specific entity based on nearby words in the embedding space.
    
    Query parameters:
    - entity_type: Type of entity (party, politician)
    - entity_name: Name of entity
    - top_n: Number of words to include (default: 100)
    - max_distance: Maximum distance threshold (default: 0.5)
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
    max_distance = float(request.args.get('max_distance', 0.5))
    
    if not entity_type or not entity_name:
        return jsonify({
            'error': 'Missing parameters',
            'message': 'Both entity_type and entity_name must be provided'
        }), 400
    
    # Get entity embedding
    entity_key = f"{entity_type}:{entity_name}"
    if entity_key not in DATA['embeddings']:
        return jsonify({
            'error': 'Entity not found',
            'message': f'No embedding found for {entity_type} {entity_name}'
        }), 404
    
    query_embedding = DATA['embeddings'][entity_key]
    
    # Find nearest words
    nearest_words = EMBEDDING_STORE.get_nearest_words(
        query_embedding, 
        top_n=top_n,
        max_distance=max_distance
    )
    
    # Format for word cloud
    word_cloud_data = [
        {
            'text': w['word'],
            'value': 1.0 - w['distance'],  # Convert distance to similarity
            'distance': w['distance']
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
