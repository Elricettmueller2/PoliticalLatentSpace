# Political Movement Visualization: Enhanced Concept

## Project Vision
Create an interactive visualization system that maps political movements in a meaningful semantic space, revealing both differences and potential overlaps through a hybrid approach combining AI and expert knowledge.

## Core Components

### 1. Hybrid Positioning System
- **Combine unsupervised embeddings with expert-defined anchors**
- **Key political dimensions:**
  - Economic: State intervention vs. Free market
  - Social: Progressive vs. Conservative
  - Environmental: Ecological priority vs. Economic growth priority
  - Governance: Nationalist vs. Internationalist
- **Implementation approach:**
  - Define clear anchor texts for each pole of each dimension
  - Calculate similarity of movement texts to these anchors
  - Position movements along multiple meaningful axes

### 2. Enhanced Data Collection
- **Structured corpus for each movement:**
  - Official manifestos/programs (1-2 documents)
  - Position papers on specific issues (3-5 documents)
  - Public statements by leadership (5-10 statements)
  - Total: ~2000-5000 words per movement
- **Issue-specific text extraction:**
  - Economy: taxation, welfare, regulation, labor
  - Environment: climate policy, energy, conservation
  - Migration: borders, integration, asylum
  - Social: healthcare, education, family policy
  - Democracy: voting rights, participation, institutions
- **Temporal dimension:**
  - Archive historical positions (past 5-10 years)
  - Track evolution of rhetoric and priorities

### 3. Multi-dimensional Visualization Approaches
- **Political Compass Plus:**
  - Enhanced 2D visualization with additional dimensions shown through size, color, and shape
  - Expert-validated axes with clear meaning
- **Issue Constellation:**
  - Movements represented as constellations across issue-specific positions
  - Lines connect a movement's positions across different issues
- **Affinity Network:**
  - Network graph where edge thickness represents agreement on specific issues
  - Filterable by issue area to show different coalition possibilities
- **Radar/Spider Charts:**
  - Multi-dimensional comparison of movement profiles
  - Overlay capability to highlight similarities/differences

### 4. Interactive Exploration Tools
- **Factor Explorer:**
  - Interactive controls to highlight specific factors influencing position
  - Ability to weight different issues by importance
- **Evidence Panel:**
  - Click on any movement to see text excerpts that define its position
  - Highlight key terms and phrases with their contribution weight
- **Time Slider:**
  - Track position changes over time
  - Animate movement trajectories in political space
- **Issue Filter:**
  - Recalculate positions based on specific issue areas
  - Compare issue-specific alignments vs. overall positioning

## Technical Implementation

### Core Technologies
- **Embedding:** Sentence-Transformers with multilingual model
- **Dimensionality Reduction:** UMAP for preserving local and global structure
- **Visualization:** Plotly for interactive web-based visualization
- **Backend:** Python with Flask or FastAPI
- **Frontend:** Simple HTML/CSS/JS interface or Streamlit

### Code Structure
```
political_space/
├── data/
│   ├── raw_texts/            # Original movement texts
│   ├── processed_texts/      # Processed and categorized texts
│   └── anchor_definitions/   # Expert-defined political anchors
├── src/
│   ├── data_processing.py    # Text processing and feature extraction
│   ├── hybrid_positioning.py # Combined embedding + expert positioning
│   ├── visualization.py      # Visualization generation
│   └── app.py               # Interactive web application
└── notebooks/
    ├── data_exploration.ipynb
    ├── model_development.ipynb
    └── visualization_design.ipynb
```

### Sample Implementation (Hybrid Positioning)
```python
class HybridPoliticalSpace:
    def __init__(self):
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        
        # Expert-defined anchor points for key political dimensions
        self.anchors = {
            "economic_left": "Wir setzen uns für staatliche Regulierung, Umverteilung und soziale Absicherung ein.",
            "economic_right": "Wir fördern freie Märkte, niedrige Steuern und unternehmerische Freiheit.",
            "progressive": "Wir stehen für gesellschaftlichen Wandel, Diversität und neue Lebensformen.",
            "conservative": "Wir bewahren Traditionen, Familie und kulturelle Identität.",
            "ecological": "Klimaschutz und Umweltschutz haben für uns höchste Priorität.",
            "nationalist": "Nationale Souveränität und Identität müssen geschützt werden.",
            "internationalist": "Internationale Zusammenarbeit und globale Lösungen sind entscheidend."
        }
        
        # Generate embeddings for anchors
        self.anchor_embeddings = {k: self.model.encode(v) for k, v in self.anchors.items()}
        
    def position_movement(self, text):
        # Get embedding for movement text
        embedding = self.model.encode(text)
        
        # Calculate similarity to each anchor
        similarities = {dim: cosine_similarity([embedding], [anchor_emb])[0][0] 
                       for dim, anchor_emb in self.anchor_embeddings.items()}
        
        # Calculate position on key axes
        economic_axis = similarities["economic_right"] - similarities["economic_left"]
        social_axis = similarities["progressive"] - similarities["conservative"]
        ecological_axis = similarities["ecological"]
        governance_axis = similarities["nationalist"] - similarities["internationalist"]
        
        return {
            "economic_axis": economic_axis,
            "social_axis": social_axis,
            "ecological_axis": ecological_axis,
            "governance_axis": governance_axis,
            "raw_embedding": embedding,
            "anchor_similarities": similarities
        }
```

## Development Roadmap

### Phase 1: Foundation (1-2 weeks)
- Set up project structure
- Implement data collection framework
- Create basic embedding pipeline
- Define expert anchor points

### Phase 2: Core Functionality (2-3 weeks)
- Implement hybrid positioning system
- Develop basic visualizations
- Create initial interactive controls
- Test with sample movement data

### Phase 3: Enhancement (2-3 weeks)
- Add temporal dimension
- Implement multiple visualization types
- Enhance interactive exploration tools
- Optimize performance

### Phase 4: Refinement (1-2 weeks)
- User testing and feedback
- Refine visualizations and interactions
- Documentation and deployment
- Prepare for public release

## Artistic Considerations
- Use color theory to enhance meaning (not just for decoration)
- Consider using sound or motion to represent additional dimensions
- Explore metaphors beyond traditional political spectrums (landscapes, constellations, ecosystems)
- Balance technical accuracy with intuitive understanding
- Create an experience that invites exploration and discovery
