import argparse
import yaml
import os
from datetime import datetime
from core.embedding import EnhancedEmbedder
from core.context_analyzer import ContextWindowAnalyzer
from core.latent_space import PoliticalLatentSpace
from core.term_analyzer import TermUsageAnalyzer
from data.loader import PoliticalTextLoader
from data.preprocessor import TextPreprocessor
from data.exporter import DataExporter
from analysis.comparative import ComparativeAnalyzer

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Political Latent Space Generator')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    embedder = EnhancedEmbedder(model_name=config['embedding']['model_name'])
    context_analyzer = ContextWindowAnalyzer(embedder, window_sizes=config['analysis']['window_sizes'])
    latent_space = PoliticalLatentSpace(embedder)
    term_analyzer = TermUsageAnalyzer(context_analyzer, latent_space)
    
    # Load and preprocess data
    loader = PoliticalTextLoader(config['data']['base_dir'])
    preprocessor = TextPreprocessor()
    
    # Load movement texts
    print("Loading movement texts...")
    movement_texts_dict = loader.load_movement_texts()
    preprocessed_movement_texts = preprocessor.preprocess_texts(movement_texts_dict)
    
    # Load politician texts
    print("Loading politician texts...")
    politician_texts_dict = loader.load_politician_texts()
    preprocessed_politician_texts = preprocessor.preprocess_texts(politician_texts_dict)
    
    # Define anchors for the latent space
    latent_space.define_anchors(config['latent_space']['anchors'])
    
    # Process movement texts
    print("\nProcessing political movements...")
    movement_results = {}
    for name, text in preprocessed_movement_texts.items():
        print(f"Processing movement: {name}...")
        position = latent_space.position_text(text)
        key_terms = term_analyzer.extract_key_terms(text)
        term_analyses = {}
        for term, _ in key_terms[:10]:  # Analyze top 10 terms
            term_analyses[term] = term_analyzer.analyze_term_usage(text, term)
        
        # Calculate position in expert dimensions
        expert_dimensions = {}
        for dimension, anchors in config['latent_space'].get('expert_dimensions', {}).items():
            if len(anchors) == 2:  # Need exactly 2 anchors for a dimension
                pos = latent_space.position_on_axis(text, anchors[0], anchors[1])
                expert_dimensions[dimension] = pos
        
        movement_results[name] = {
            'position': position,
            'key_terms': key_terms,
            'term_analyses': term_analyses,
            'expert_dimensions': expert_dimensions,
            'embedding': embedder.encode(text).tolist()  # Store the full embedding
        }
    
    # Process politician texts
    print("\nProcessing individual politicians...")
    politician_results = {}
    for name, data in preprocessed_politician_texts.items():
        text = data['text']
        movement = data['movement']
        print(f"Processing politician: {name} ({movement})...")
        
        position = latent_space.position_text(text)
        key_terms = term_analyzer.extract_key_terms(text)
        term_analyses = {}
        for term, _ in key_terms[:5]:  # Analyze top 5 terms for politicians
            term_analyses[term] = term_analyzer.analyze_term_usage(text, term)
        
        # Calculate position in expert dimensions
        expert_dimensions = {}
        for dimension, anchors in config['latent_space'].get('expert_dimensions', {}).items():
            if len(anchors) == 2:  # Need exactly 2 anchors for a dimension
                pos = latent_space.position_on_axis(text, anchors[0], anchors[1])
                expert_dimensions[dimension] = pos
        
        politician_results[name] = {
            'position': position,
            'key_terms': key_terms,
            'term_analyses': term_analyses,
            'expert_dimensions': expert_dimensions,
            'movement': movement,
            'embedding': embedder.encode(text).tolist()  # Store the full embedding
        }
    
    # Perform comparative analysis
    print("\nPerforming comparative analysis...")
    comparative = ComparativeAnalyzer(latent_space, term_analyzer)
    movement_comparison = comparative.compare_movements(preprocessed_movement_texts)
    
    # Export results
    print("\nExporting results...")
    exporter = DataExporter(config['output']['dir'])
    exporter.export_to_json({
        'movements': movement_results,
        'politicians': politician_results,
        'comparison': movement_comparison,
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'model': config['embedding']['model_name'],
            'includes_politicians': True
        }
    }, config['output']['filename'])
    
    print(f"Analysis complete. Results saved to {os.path.join(config['output']['dir'], config['output']['filename'])}")

if __name__ == '__main__':
    main()