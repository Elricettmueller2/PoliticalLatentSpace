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

# Use absolute paths based on the location of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_RESULTS_PATH = os.path.join(BASE_DIR, 'src/data/processed/multi_level_analysis_results.json')
WORD_EMBEDDING_PATH = os.path.join(BASE_DIR, 'src/data/processed/word_embeddings.h5')
WORD_INDEX_PATH = os.path.join(BASE_DIR, 'src/data/processed/word_embeddings.index')

# Enhanced debugging for cross-platform compatibility
print(f"Current working directory: {os.getcwd()}")
print(f"Application directory: {BASE_DIR}")
print(f"Checking file existence and sizes:")
for path in [ANALYSIS_RESULTS_PATH, WORD_EMBEDDING_PATH, WORD_INDEX_PATH]:
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"  - {path}: EXISTS, size={size/1024/1024:.2f} MB")
    else:
        print(f"  - {path}: NOT FOUND")

# Load the data directly from the JSON file
DATA = {}
try:
    print(f"Loading data directly from {ANALYSIS_RESULTS_PATH}...")
    with open(ANALYSIS_RESULTS_PATH, 'r', encoding='utf-8') as f:
        DATA = json.load(f)
    print(f"Data loaded successfully with keys: {list(DATA.keys())}")
    if 'movements' in DATA:
        print(f"Found {len(DATA['movements'])} movements")
        # Debug: Check first movement for embedding
        first_movement = next(iter(DATA['movements'].items()))
        print(f"First movement: {first_movement[0]}")
        print(f"Has embedding: {'embedding' in first_movement[1]}")
        if 'embedding' in first_movement[1]:
            print(f"Embedding type: {type(first_movement[1]['embedding'])}")
            print(f"Embedding length: {len(first_movement[1]['embedding'])}")
            
    if 'politicians' in DATA:
        print(f"Found {len(DATA['politicians'])} politicians")
        # Debug: Check first politician for embedding
        first_politician = next(iter(DATA['politicians'].items()))
        print(f"First politician: {first_politician[0]}")
        print(f"Has embedding: {'embedding' in first_politician[1]}")
        if 'embedding' in first_politician[1]:
            print(f"Embedding type: {type(first_politician[1]['embedding'])}")
            print(f"Embedding length: {len(first_politician[1]['embedding'])}")
    
    # Initialize empty embeddings dictionary if needed
    if 'embeddings' not in DATA:
        DATA['embeddings'] = {}
        
except Exception as e:
    print(f"Error loading data: {e}")
    import traceback
    traceback.print_exc()
    DATA = {'movements': {}, 'politicians': {}, 'embeddings': {}}

# Initialize embedding store and latent space
EMBEDDING_STORE = None
LATENT_SPACE = None

try:
    # Check if word embeddings file exists
    if os.path.exists(WORD_EMBEDDING_PATH) and os.path.exists(WORD_INDEX_PATH):
        print(f"Loading word embeddings from {WORD_EMBEDDING_PATH}...")
        try:
            import h5py
            with h5py.File(WORD_EMBEDDING_PATH, 'r') as f:
                print(f"HDF5 file structure: {list(f.keys())}")
                if 'words' in f:
                    print(f"Words dataset shape: {f['words'].shape}")
                if 'embeddings' in f:
                    print(f"Embeddings dataset shape: {f['embeddings'].shape}")
        except Exception as e:
            print(f"Error inspecting HDF5 file: {e}")
            traceback.print_exc()
            
        try:
            import faiss
            index = faiss.read_index(WORD_INDEX_PATH)
            print(f"FAISS index loaded successfully, dimension: {index.d}, ntotal: {index.ntotal}")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            traceback.print_exc()
            
        EMBEDDING_STORE = ChunkedEmbeddingStore(
            WORD_EMBEDDING_PATH,
            index_file=WORD_INDEX_PATH,
            cache_size=1000,
            verbose=True
        )
        print("Word embeddings loaded successfully")
        
        # Initialize latent space
        print("Initializing latent space...")
        LATENT_SPACE = MultiLevelLatentSpace(
            entity_data_path=ANALYSIS_RESULTS_PATH,
            word_embedding_path=WORD_EMBEDDING_PATH,
            index_file=WORD_INDEX_PATH,
            verbose=True
        )
        print("Latent space initialized successfully")
        
        # Check if entities have embeddings
        print("Checking entity embeddings...")
        sample_movement = next(iter(DATA.get('movements', {}).keys()), None)
        sample_politician = next(iter(DATA.get('politicians', {}).keys()), None)
        
        if sample_movement:
            movement_embedding = LATENT_SPACE.get_entity_embedding('movement', sample_movement)
            print(f"Sample movement '{sample_movement}' has embedding: {movement_embedding is not None}")
            
        if sample_politician:
            politician_embedding = LATENT_SPACE.get_entity_embedding('politician', sample_politician)
            print(f"Sample politician '{sample_politician}' has embedding: {politician_embedding is not None}")
            
        # Populate DATA['embeddings'] with entity embeddings from the latent space
        print("Populating embeddings dictionary...")
        # Add movement embeddings
        for movement_name in DATA.get('movements', {}):
            embedding = LATENT_SPACE.get_entity_embedding('movement', movement_name)
            if embedding is not None:
                DATA['embeddings'][f"movement:{movement_name}"] = embedding
                
        # Add politician embeddings
        for politician_name in DATA.get('politicians', {}):
            embedding = LATENT_SPACE.get_entity_embedding('politician', politician_name)
            if embedding is not None:
                DATA['embeddings'][f"politician:{politician_name}"] = embedding
                
        print(f"Populated embeddings dictionary with {len(DATA['embeddings'])} entity embeddings")
    else:
        print(f"Word embeddings file not found at {WORD_EMBEDDING_PATH}")
        print(f"Index file not found at {WORD_INDEX_PATH}")
        print("Running with fallback word clouds only")
except Exception as e:
    print(f"Error initializing embedding store: {e}")
    import traceback
    traceback.print_exc()
    print("Running with fallback word clouds only")

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

@app.route('/api/entity-focus')
def get_entity_focus():
    """Provide data for the 2D drill-down visualization that focuses on a selected entity and its relationships."""
    # Get parameters
    entity_type = request.args.get('entity_type')
    entity_name = request.args.get('entity_name')
    num_words = int(request.args.get('num_words', 50))
    
    print(f"\n--- Entity Focus Request: {entity_type} {entity_name} ---")
    
    if not entity_type or not entity_name:
        return jsonify({
            'error': True,
            'message': 'Entity type and name are required for entity focus view'
        })
    
    try:
        # Get entity data
        entity_data = None
        if entity_type == 'movement' and entity_name in DATA.get('movements', {}).keys():
            entity_data = DATA['movements'][entity_name]
        elif entity_type == 'politician' and entity_name in DATA.get('politicians', {}).keys():
            entity_data = DATA['politicians'][entity_name]
        
        if not entity_data:
            return jsonify({
                'error': True,
                'message': f'Entity not found: {entity_type} {entity_name}'
            })
        
        # Get related entities
        related_entities = []
        
        # For movements, get related politicians
        if entity_type == 'movement':
            for politician_name, politician_data in DATA['politicians'].items():
                if politician_data.get('movement') == entity_name:
                    # Calculate similarity or use position data
                    position = politician_data.get('position', {}).get('expert_dimensions', {}).get('axes', {})
                    related_entities.append({
                        'name': politician_name,
                        'type': 'politician',
                        'movement': entity_name,
                        'position': position,
                        'similarity': 0.8  # Placeholder, will calculate actual similarity
                    })
            
            # Add other movements based on similarity
            for other_movement, movement_data in DATA['movements'].items():
                if other_movement != entity_name:
                    # Calculate similarity between movements
                    position = movement_data.get('position', {}).get('expert_dimensions', {}).get('axes', {})
                    # Simple similarity based on position differences
                    similarity = 0.5  # Placeholder
                    related_entities.append({
                        'name': other_movement,
                        'type': 'movement',
                        'position': position,
                        'similarity': similarity
                    })
        
        # For politicians, get their movement and colleagues
        elif entity_type == 'politician':
            # Get politician's movement
            politician_movement = entity_data.get('movement')
            if politician_movement and politician_movement in DATA['movements']:
                movement_data = DATA['movements'][politician_movement]
                position = movement_data.get('position', {}).get('expert_dimensions', {}).get('axes', {})
                related_entities.append({
                    'name': politician_movement,
                    'type': 'movement',
                    'position': position,
                    'similarity': 0.9  # High similarity to own movement
                })
                
                # Get colleagues from same movement
                for politician_name, politician_data in DATA['politicians'].items():
                    if politician_data.get('movement') == politician_movement and politician_name != entity_name:
                        position = politician_data.get('position', {}).get('expert_dimensions', {}).get('axes', {})
                        related_entities.append({
                            'name': politician_name,
                            'type': 'politician',
                            'movement': politician_movement,
                            'position': position,
                            'similarity': 0.7  # Placeholder for colleague similarity
                        })
        
        # Get word cloud data - Try to use embeddings first, fall back to static data if needed
        word_cloud_viz = []
        
        # Try to get word cloud from latent space first
        if LATENT_SPACE and EMBEDDING_STORE:
            try:
                print(f"Attempting to get word cloud for {entity_type} {entity_name} from embeddings...")
                
                # Debug: Check entity embedding
                entity_embedding = LATENT_SPACE.get_entity_embedding(entity_type, entity_name)
                if entity_embedding is not None:
                    print(f"Entity embedding shape: {entity_embedding.shape}")
                    print(f"Entity embedding norm: {np.linalg.norm(entity_embedding)}")
                    print(f"Entity embedding sample: {entity_embedding[:5]}")
                    
                    # Debug: Check a few word embeddings
                    if EMBEDDING_STORE:
                        print("Checking word embeddings...")
                        sample_words = ["politik", "demokratie", "wirtschaft"]
                        for word in sample_words:
                            word_emb = EMBEDDING_STORE.get_word_embedding(word)
                            if word_emb is not None:
                                print(f"Word '{word}' embedding norm: {np.linalg.norm(word_emb)}")
                                # Calculate cosine similarity
                                similarity = np.dot(entity_embedding, word_emb) / (np.linalg.norm(entity_embedding) * np.linalg.norm(word_emb))
                                print(f"Similarity to '{entity_name}': {similarity}")
                            else:
                                print(f"No embedding for word '{word}'")
                
                # Use the dedicated word cloud method with very high max_distance
                word_cloud_data = LATENT_SPACE.get_word_cloud_for_entity(
                    entity_type,
                    entity_name,
                    top_n=num_words,
                    max_distance=10.0  # Very high to see if we get any results
                )
                
                print(f"Found {len(word_cloud_data)} words for word cloud")
                if len(word_cloud_data) > 0:
                    print(f"First few words: {[w['text'] for w in word_cloud_data[:5]]}")
                
                # Format for visualization
                for word_item in word_cloud_data:
                    word = word_item['text']
                    similarity = word_item['value']  # Already converted from distance
                    
                    # Get position for the term if available in WORD_DIMENSION_SCORES
                    position = WORD_DIMENSION_SCORES.get(word.lower(), [0, 0, 0])
                    
                    word_cloud_viz.append({
                        'text': word,
                        'value': similarity,
                        'position': position
                    })
                
                print(f"Generated word cloud with {len(word_cloud_viz)} words from embeddings")
            except Exception as e:
                print(f"Error generating word cloud from embeddings: {e}")
                import traceback
                traceback.print_exc()
                # Will fall back to static data
        
        # If no words were found using embeddings, use fallback method
        if not word_cloud_viz:
            print("Using fallback word cloud generation")
            word_cloud = entity_data.get('word_cloud', {})
            if not word_cloud:  # If word cloud is empty, generate sample data
                word_cloud = generate_word_cloud(entity_type, entity_name)
            
            # Format word cloud for visualization
            for term, weight in word_cloud.items():
                # Get position for the term if available in WORD_DIMENSION_SCORES
                position = WORD_DIMENSION_SCORES.get(term.lower(), [0, 0, 0])
                word_cloud_viz.append({
                    'text': term,
                    'value': weight,
                    'position': position
                })
        
        # Sort by value for importance
        word_cloud_viz.sort(key=lambda x: x['value'], reverse=True)
        
        # Limit to top words
        word_cloud_viz = word_cloud_viz[:num_words]
        
        # Create 2D visualization data
        focus_data = {
            'entity': {
                'name': entity_name,
                'type': entity_type,
                'position': entity_data.get('position', {}).get('expert_dimensions', {}).get('axes', {})
            },
            'related_entities': related_entities,
            'word_cloud': word_cloud_viz
        }
        
        return jsonify(focus_data)
        
    except Exception as e:
        print(f"Error generating entity focus view: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a valid JSON error response
        return jsonify({
            'error': True,
            'message': f"Failed to generate entity focus view: {str(e)}"
        })

@app.route('/api/hybrid-visualization')
def get_hybrid_visualization():
    """Provide data for the hybrid visualization that focuses on a selected entity and its relationships."""
    # Get parameters
    entity_type = request.args.get('entity_type')
    entity_name = request.args.get('entity_name')
    num_words = int(request.args.get('num_words', 50))
    
    if not entity_type or not entity_name:
        return jsonify({
            'error': True,
            'message': 'Entity type and name are required for hybrid visualization'
        })
    
    try:
        # Create a selected entity object
        selected_entity = {'type': entity_type, 'name': entity_name}
        
        # Use the same visualization function but with a flag to indicate hybrid mode
        # This could be extended in the future to use a specialized function
        galaxy_fig = create_galaxy_visualization(
            DATA, 
            selected_entity=selected_entity,
            hybrid_mode=True,  # Flag to indicate hybrid visualization mode
            num_words=num_words
        )
        
        # Use Plotly's JSON encoder to serialize the figure
        return json.dumps(galaxy_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
    except Exception as e:
        print(f"Error generating hybrid visualization: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a valid JSON error response
        error_response = {
            'error': True,
            'message': f"Failed to generate hybrid visualization: {str(e)}",
            'data': [],
            'layout': {
                'title': 'Error: Could not load hybrid visualization',
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

# Entity info endpoint is already defined at line 347

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
    nearest_words = LATENT_SPACE.get_word_cloud_for_entity(
        entity_type,
        entity_name,
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
    nearest_words = LATENT_SPACE.get_word_cloud_for_entity(
        entity_type,
        entity_name,
        top_n=top_n,
        max_distance=max_distance
    )
    
    # Format for word cloud
    word_cloud_data = [
        {
            'text': w['text'],
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
