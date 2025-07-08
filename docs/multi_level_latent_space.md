# Multi-Level Political Latent Space Documentation

## Overview

The Multi-Level Political Latent Space is a hybrid approach that combines:

1. **Entity-level embeddings** - Compact, interpretable representations of political entities (parties, politicians)
2. **Word-level embeddings** - Detailed, rich embeddings for individual words in political texts

This hybrid approach enables both high-level political landscape visualization and detailed semantic exploration with drill-down capabilities, allowing users to understand both the macro-level political positioning and the micro-level language patterns that define political entities.

## Data Structure

### Entity-Level Data (`multi_level_analysis_results.json`)

The entity-level data is stored in a JSON file with the following structure:

```json
{
  "movements": {
    "movement_name": {
      "text": "preprocessed text content",
      "embedding": [float array],
      "position": {
        "coordinates": {"x": float, "y": float},
        "size": float,
        "color": "rgba string",
        "expert_dimensions": {
          "axes": {
            "economic": float,
            "social": float,
            "ecological": float
          }
        }
      }
    },
    ...
  },
  "politicians": {
    "politician_name": {
      "text": "preprocessed text content",
      "movement": "movement_name",
      "embedding": [float array],
      "position": {
        "coordinates": {"x": float, "y": float},
        "size": float,
        "color": "rgba string",
        "expert_dimensions": {
          "axes": {
            "economic": float,
            "social": float,
            "ecological": float
          }
        }
      }
    },
    ...
  }
}
```

### Word-Level Embeddings (`word_embeddings.h5` and `.index`)

Word embeddings are stored in two files:

1. **HDF5 File** (`word_embeddings.h5`): Contains the actual embedding vectors for each word
2. **FAISS Index** (`word_embeddings.index`): Enables efficient similarity search

The word embeddings are managed by the `ChunkedEmbeddingStore` class, which provides efficient access and caching.

## Key Features

### 1. Entity Management

- Retrieve all political movements and politicians
- Get detailed information about specific entities
- Access entity embeddings for custom analysis

### 2. Word Embedding Operations

- Check if a word has an embedding
- Retrieve word embeddings
- Find similar words based on embedding similarity

### 3. Entity-Word Relationships

- Find nearest words to a given entity
- Generate word clouds for entities
- Explore the semantic neighborhood of political entities

### 4. Hybrid Visualization

- Generate data for visualizing both entities and words in the same space
- Support for interactive drill-down from entities to related words
- Customizable visualization parameters

## Usage

### Loading the Latent Space

```python
from src.data.embeddings.multi_level_latent_space import MultiLevelLatentSpace

# Load existing latent space
latent_space = MultiLevelLatentSpace(
    entity_file='src/data/processed/multi_level_analysis_results.json',
    word_embedding_file='src/data/processed/word_embeddings.h5',
    verbose=True
)
```

### Creating a New Latent Space

```python
# Create a new latent space from raw data
latent_space = MultiLevelLatentSpace(verbose=True)
success = latent_space.create_new_latent_space(
    entity_data=entity_data,
    word_embeddings=word_embeddings,
    output_dir='src/data/processed',
    entity_output_file='multi_level_analysis_results.json',
    word_embedding_output_file='word_embeddings.h5',
    index_type='flat'  # Options: 'flat', 'ivf', 'hnsw'
)
```

### Entity Operations

```python
# Get all movements
movements = latent_space.get_all_movements()

# Get all politicians
politicians = latent_space.get_all_politicians()

# Get entity information
entity_info = latent_space.get_entity_info('CDU')

# Get entity embedding
embedding = latent_space.get_entity_embedding('Angela Merkel')
```

### Word Operations

```python
# Check if a word has an embedding
has_embedding = latent_space.has_word_embedding('politik')

# Get word embedding
embedding = latent_space.get_word_embedding('wirtschaft')

# Find similar words
similar_words = latent_space.find_similar_words('umwelt', k=10)
```

### Entity-Word Relationships

```python
# Find nearest words to an entity
nearest_words = latent_space.find_nearest_words_to_entity('Gruene', k=20)

# Generate word cloud for an entity
word_cloud = latent_space.generate_word_cloud('SPD', max_words=50)
```

### Hybrid Visualization

```python
# Generate hybrid visualization data
visualization_data = latent_space.generate_hybrid_visualization_data(
    num_words=200,
    min_word_freq=5,
    entity_types=['movements', 'politicians']
)
```

## Scripts

### Building the Latent Space

The `build_multi_level_latent_space.py` script creates a new latent space from raw political texts:

```bash
python scripts/build_multi_level_latent_space.py \
    --raw-dir src/data/raw \
    --output-dir src/data/processed \
    --entity-file multi_level_analysis_results.json \
    --embedding-file word_embeddings.h5 \
    --embedding-dim 300 \
    --index-type flat
```

### Testing the Latent Space

The `test_multi_level_latent_space.py` script validates the latent space implementation:

```bash
python scripts/test_multi_level_latent_space.py \
    --data-dir src/data/processed \
    --entity-file multi_level_analysis_results.json \
    --embedding-file word_embeddings.h5 \
    --verbose
```

## Integration with Backend

The multi-level latent space can be integrated with the Flask backend by updating the `app.py` file to use the `MultiLevelLatentSpace` class for serving data to the frontend.

Example endpoints:

1. `/api/galaxy` - Entity-level visualization data
2. `/api/entity/<entity_id>` - Detailed entity information
3. `/api/nearest_words/<entity_id>` - Words most related to an entity
4. `/api/hybrid_visualization` - Combined entity and word visualization data
5. `/api/word_cloud/<entity_id>` - Word cloud data for an entity

## Performance Considerations

### Memory Usage

The word embeddings can be large (multi-GB), but the `ChunkedEmbeddingStore` provides efficient memory management through:

- Chunked storage in HDF5 format
- FAISS indexing for fast similarity search
- LRU caching for frequently accessed embeddings

### Scaling

For large-scale deployments:

1. Use more efficient FAISS index types ('ivf' or 'hnsw')
2. Adjust cache size based on available memory
3. Consider using a smaller test dataset for development

## Future Extensions

1. **Dynamic dimensionality reduction**: Project word embeddings into the same 2D/3D space as entities on-demand
2. **Temporal analysis**: Track how entities move in the latent space over time
3. **Cross-lingual support**: Extend to multiple languages with aligned embedding spaces
4. **Interactive clustering**: Group entities and words based on user-defined criteria
5. **Personalized views**: Allow users to define their own axes and projections
