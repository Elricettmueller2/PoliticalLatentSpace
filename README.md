# Political Latent Space Analysis Tool

This project provides a comprehensive suite of tools for analyzing and visualizing political texts. It allows users to explore the ideological positions of political parties, movements, and individual politicians by embedding their texts into a high-dimensional latent space. The application supports multi-level analysis, contextual term analysis, and the interpretation of learned ideological dimensions.

## Features

- **Latent Space Analysis**: Maps political texts onto a latent space defined by expert-knowledge anchors (e.g., economic left/right) and data-driven dimensions learned through techniques like UMAP, PCA, or t-SNE.
- **Multi-Level Analysis**: Supports hierarchical analysis, allowing for the comparison of political parties and individual politicians within a unified framework. This includes features like party cohesion analysis and outlier detection.
- **Term and Context Analysis**: Provides tools for extracting key terms from texts (using TF-IDF) and analyzing how specific terms are used in different contexts. This allows for a deeper understanding of political language and framing.
- **Dimension Interpretation**: Offers utilities for interpreting the meaning of learned latent dimensions by correlating them with reference texts and identifying exemplar texts for each dimension.
- **Interactive Visualization**: A Flask-based web application with a Plotly frontend allows for the interactive exploration of the political latent space.
- **Configurable Pipeline**: The entire analysis pipeline is configurable through a central `config.yaml` file, allowing users to easily customize embedding models, analysis parameters, and data paths.

## Architecture

The project is composed of a Python backend for data processing and analysis, and a JavaScript-based frontend for visualization.

- **Backend**:
  - **Flask Application (`app.py`)**: Serves the frontend and provides a REST API for accessing analysis results.
  - **Analysis Pipeline (`src/main.py`)**: Orchestrates the data loading, preprocessing, analysis, and exporting workflow.
  - **Core Modules (`src/core`)**: Contains the analytical engine of the project, including modules for embedding, latent space creation, term analysis, and dimension interpretation.
  - **Data Pipeline (`src/data`)**: Manages the loading, preprocessing, and exporting of data.

- **Frontend**:
  - **HTML/JavaScript/CSS (`templates/` and `static/`)**: Provides an interactive interface for visualizing the political latent space.


## Configuration

```bash
python -m venv venv
./venv/Scripts/activate
pip install -r requirements.txt
```

The analysis pipeline is configured through the `config.yaml` file. Here you can define:

- **`embedding_model`**: The sentence-transformer model to use for text embeddings.
- **`analysis`**: Parameters for the analysis, such as context window sizes and the number of top terms to extract.
- **`paths`**: The input and output directories for data.
- **`latent_space`**: The anchors for the expert-defined dimensions of the latent space.
- **`output`**: The filenames for the exported analysis results.

## Usage

### Running the Analysis

To run the full analysis pipeline, execute the `main.py` script:

It is not required since the latent space is already created in the `src/data/processed` directory.

```bash
python src/main.py
```

### Running the Web Application

To start the web application, run the `app.py` script:

```bash
python app.py
```

Then, open your web browser and navigate to `http://127.0.0.1:5000`.

## Core Modules

### `src/core`

- **`embedding.py`**: Defines the `EnhancedEmbedder` class for generating text embeddings using sentence-transformer models. This class provides batch processing capabilities and text segmentation for handling long documents efficiently.

- **`latent_space.py`**: Defines the `PoliticalLatentSpace` class for creating and managing the latent space. It supports both expert-defined semantic anchors and learned dimensions through dimensionality reduction techniques. Key features include:
  - Positioning texts in the latent space based on their embeddings
  - Comparing texts by their positions in the latent space
  - Evaluating the quality of learned dimensions
  - Supporting multiple dimensionality reduction methods (UMAP, PCA, t-SNE)

- **`multi_level_latent_space.py`**: Defines the `MultiLevelLatentSpace` class for handling hierarchical data (e.g., parties and politicians). It enables:
  - Building separate latent spaces for different entity levels
  - Mapping between levels (e.g., politicians to parties)
  - Detecting outliers within groups
  - Calculating group cohesion metrics
  - Comparing entities across different levels

- **`context_analyzer.py`**: Defines the `ContextWindowAnalyzer` class for analyzing how specific terms are used in context within political texts. Features include:
  - Extracting context windows around term occurrences
  - Analyzing term usage in different contexts
  - Comparing contexts across different texts
  - Finding semantic clusters to identify different usages of a term

- **`term_analyzer.py`**: Defines the `TermUsageAnalyzer` class for extracting and analyzing key terms. It provides:
  - Key term extraction using TF-IDF
  - Term usage analysis with context positioning in latent space
  - Comparative analysis of term usage across different texts
  - Finding distinctive terms that differentiate texts

- **`dimension_interpreter.py`**: Defines the `DimensionInterpreter` class for interpreting the learned dimensions of the latent space. It offers:
  - Correlation of dimensions with reference concepts
  - Calculation of dimension importance based on variance explained
  - Identification of exemplar texts for each dimension
  - Analysis of dimension redundancy through correlation matrices

### `src/data`

- **`loader.py`**: Defines the `PoliticalTextLoader` class for loading raw text and PDF data. It supports:
  - Loading texts for political movements from directories
  - Loading texts for individual politicians organized by movements
  - Handling various file formats (TXT, PDF)
  - Extracting text from PDF files

- **`preprocessor.py`**: Defines the `TextPreprocessor` class for cleaning, normalizing, and segmenting text data. Features include:
  - Text cleaning and normalization
  - Segmentation of long texts into smaller chunks
  - Intelligent overlap point selection for segmentation
  - Support for custom preprocessing functions

- **`exporter.py`**: Defines the `DataExporter` class for saving processed data and analysis results. It supports:
  - Exporting to JSON format
  - Exporting to CSV format
  - Specialized formatting for visualization data
  - Exporting term analysis results

## Prototypes

The `prototypes/` directory contains experimental implementations and visualization concepts that were developed during the project's early stages. These serve as proof-of-concept implementations and exploratory analyses. 

They either create a png or html file that visualizes the latent space and allow simple interpretation.

### Key Prototype Files

- **`political_space_prototype.py`**: A prototype for visualizing political movements in a shared embedding space. It demonstrates:
  - Basic embedding generation using sentence-transformers
  - Dimensionality reduction with UMAP
  - Network visualization of political movements based on text similarity
  - Interactive visualization using Plotly

- **`galaxy_visualization.py`**: An advanced visualization prototype that creates a "galaxy" view of the political space, with:
  - Interactive 3D visualization of political entities
  - Clustering of similar entities into "solar systems"
  - Dynamic zooming and exploration capabilities
  - Color coding based on political dimensions

- **`hybrid_political_space.py`**: Explores combining expert-defined dimensions with learned dimensions to create a hybrid political space.

- **`latent_space_explorer.py`**: A tool for interactively exploring the latent space, allowing users to:
  - Navigate through the space
  - Query nearest neighbors to specific points
  - Visualize term relationships
  - Project new texts into the existing space

- **`radar_chart_visualization.py`**: Creates radar charts to visualize political entities along multiple dimensions simultaneously.

- **`vector_space_interpreter.py`**: Early implementation of tools to interpret the meaning of vectors in the embedding space.

## Scripts

The `scripts/` directory contains utility scripts for specific tasks related to the analysis pipeline. These scripts can be run independently for targeted analyses.

### Key Script Files

- **`analyze_latent_space.py`**: Provides utilities to work directly with the Political Latent Space, focusing on analyzing word clouds and their relationships to political entities. Features include:
  - Word cloud generation for specific entities
  - Comparison of word clouds between entities
  - Finding distinctive words for entities
  - Analyzing political dimensions based on word clouds

- **`build_multi_level_latent_space.py`**: Creates a multi-level latent space from raw data, supporting both party and politician levels.

- **`compare_word_clouds.py`**: Specialized tool for detailed comparison of word clouds between political entities.

- **`entity_term_analysis.py`**: Analyzes how specific terms are used by different political entities, identifying patterns in language usage.

- **`explore_latent_space.py`**: Interactive command-line tool for exploring the latent space, querying entities, and analyzing relationships.

- **`build_index.py`**: Creates a FAISS index for efficient similarity search in the embedding space.

- **`download_embeddings.py`**: Utility for downloading pre-trained embedding models.

## Data Structure

The project uses the following data structure:

- **`src/data/raw/`**: Contains raw text data organized by political movements and politicians.
- **`src/data/processed/`**: Stores processed data, including embeddings, analysis results, and visualization data.
- **`src/data/embeddings/`**: Contains embedding-related modules and cached embeddings.

## Visualization Features

The web application provides several visualization features:

- **Galaxy View**: A 2D or 3D visualization of political entities positioned in the latent space.
- **Entity Focus**: Detailed view of a specific entity, showing its position, key terms, and related entities.
- **Word Cloud**: Visual representation of terms associated with an entity, sized by relevance.
- **Comparative Analysis**: Tools for comparing multiple entities along different dimensions.
- **Dimension Explorer**: Interface for exploring the meaning of different latent dimensions.

## Future Development Plans

The project has several targeted areas for improvement:

### Current Challenges and Planned Solutions

1. **Focus View Integration**: Currently facing issues with integrating entity data and word clouds in the focus view. Future work would develop a more seamless connection between these components to better visualize how terms relate to an entity's position.

2. **Word Cloud Enhancements**: Planning to improve word cloud functionality with better term relevance scoring, more intuitive visualization, and improved filtering of non-informative terms.

3. **Latent Space Interpretation**:  Better methods to interpret the meaning of latent dimensions and regions, including more intuitive labeling and explanation of what different positions in the space represent.

4. **Visual Representation**: Enhancing the visual display of entities, relationships, and dimensions with more intuitive and informative graphics that better communicate political positioning.
