# Political Latent Space Generator

A sophisticated tool for analyzing political texts and positioning them in a multidimensional semantic space. This project processes political manifestos, speeches, and position papers to create a latent space representation that captures multiple political dimensions.

## Features

- **Enhanced Embedding Engine**: Uses multilingual sentence transformers to generate embeddings for political texts
- **Context Window Analysis**: Analyzes how specific terms are used in context within political texts
- **Multidimensional Latent Space**: Combines expert-defined political dimensions with learned dimensions
- **Term Usage Analysis**: Extracts key terms and analyzes their usage across different political movements
- **Comparative Analysis**: Compares political movements and their term usage patterns

## Project Structure

```
PoliticalLatentSpace/
├── data/
│   ├── raw/           # Raw political texts
│   └── processed/     # Processed data and results
├── src/
│   ├── core/          # Core components
│   │   ├── embedding.py
│   │   ├── context_analyzer.py
│   │   ├── latent_space.py
│   │   └── term_analyzer.py
│   ├── data/          # Data processing
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── exporter.py
│   ├── analysis/      # Analysis components
│   │   └── comparative.py
│   ├── config.yaml    # Configuration file
│   └── main.py        # Main execution script
├── prototypes/        # Prototype implementations
└── requirements.txt   # Dependencies
```

## Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure the system in `src/config.yaml`

3. Run the analysis:
   ```
   python src/main.py --config src/config.yaml
   ```

## Configuration

The system is configured via `config.yaml`, which includes:

- Embedding model settings
- Analysis parameters
- Data paths
- Political dimension anchors
- Output settings

## Requirements

- Python 3.7+
- sentence-transformers
- numpy
- pandas
- scikit-learn
- umap-learn
- plotly
- matplotlib
- PyYAML

## Future Enhancements

- Temporal analysis of political discourse
- Rhetorical analysis
- Interactive visualization dashboard
- Fine-tuning of embedding models for political domain
