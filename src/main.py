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
    texts_dict = loader.load_movement_texts()
    preprocessed_texts = preprocessor.preprocess_texts(texts_dict)
    
    # Define anchors for the latent space
    latent_space.define_anchors(config['latent_space']['anchors'])
    
    # Process texts
    results = {}
    for name, text in preprocessed_texts.items():
        print(f"Processing {name}...")
        position = latent_space.position_text(text)
        key_terms = term_analyzer.extract_key_terms(text)
        term_analyses = {}
        for term, _ in key_terms[:10]:  # Analyze top 10 terms
            term_analyses[term] = term_analyzer.analyze_term_usage(text, term)
        
        results[name] = {
            'position': position,
            'key_terms': key_terms,
            'term_analyses': term_analyses
        }
    
    # Perform comparative analysis
    comparative = ComparativeAnalyzer(latent_space, term_analyzer)
    comparison = comparative.compare_movements(preprocessed_texts)
    
    # Export results
    exporter = DataExporter(config['output']['dir'])
    exporter.export_to_json({
        'movements': results,
        'comparison': comparison,
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'model': config['embedding']['model_name']
        }
    }, config['output']['filename'])
    
    print(f"Analysis complete. Results saved to {os.path.join(config['output']['dir'], config['output']['filename'])}")

if __name__ == '__main__':
    main()