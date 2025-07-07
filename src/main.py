import argparse
import yaml
import os
from datetime import datetime
from core.embedding import EnhancedEmbedder
from core.context_analyzer import ContextWindowAnalyzer
from core.latent_space import PoliticalLatentSpace
from core.multi_level_latent_space import MultiLevelLatentSpace
from core.term_analyzer import TermUsageAnalyzer
from data.loader import PoliticalTextLoader
from data.preprocessor import TextPreprocessor
from data.exporter import DataExporter
from analysis.comparative import ComparativeAnalyzer

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Political Latent Space Generator')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--multi-level', action='store_true', help='Enable multi-level analysis of parties and politicians')
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
    
    if args.multi_level:
        # Run multi-level analysis if requested
        multi_level_results = run_multi_level_analysis(
            embedder, 
            preprocessed_movement_texts, 
            preprocessed_politician_texts, 
            config
        )
        
        # Export multi-level results
        print("\nExporting multi-level analysis results...")
        exporter = DataExporter(config['output']['dir'])
        exporter.export_to_json(multi_level_results, 'multi_level_analysis_results.json')
        
        print(f"Multi-level analysis complete. Results saved to {os.path.join(config['output']['dir'], 'multi_level_analysis_results.json')}")
        return
    
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

def run_multi_level_analysis(embedder, party_texts, politician_texts_dict, config):
    """
    Run multi-level latent space analysis on parties and politicians.
    
    Args:
        embedder: The text embedder to use
        party_texts: Dictionary of party/movement texts
        politician_texts_dict: Dictionary of politician texts with metadata
        config: Configuration dictionary
        
    Returns:
        Dictionary with analysis results
    """
    print("\n=== Running Multi-Level Political Latent Space Analysis ===")
    
    # Create multi-level latent space
    multi_space = MultiLevelLatentSpace(embedder)
    
    # Define semantic anchors
    print("Defining semantic anchors...")
    multi_space.define_anchors(config['latent_space']['anchors'])
    
    # Process politician texts to match format expected by multi-level latent space
    processed_politician_texts = {}
    politician_party_mapping = {}
    
    for name, data in politician_texts_dict.items():
        processed_politician_texts[name] = {
            'text': data['text'],
            'party': data['movement']
        }
        politician_party_mapping[name] = data['movement']
    
    # Build party latent space
    print("\nBuilding party latent space...")
    party_result = multi_space.build_party_latent_space(
        party_texts=party_texts,
        n_components=config['latent_space'].get('n_components', 4),
        method=config['latent_space'].get('method', 'umap')
    )
    
    if not party_result["success"]:
        print(f"Error building party latent space: {party_result.get('error', 'Unknown error')}")
        return {"success": False, "error": party_result.get('error')}
    
    print(f"Successfully built party latent space with {party_result['n_components']} dimensions")
    
    # Build politician latent space
    print("\nBuilding politician latent space...")
    politician_result = multi_space.build_politician_latent_space(
        politician_texts=processed_politician_texts,
        n_components=config['latent_space'].get('n_components', 4),
        method=config['latent_space'].get('method', 'umap')
    )
    
    if not politician_result["success"]:
        print(f"Error building politician latent space: {politician_result.get('error', 'Unknown error')}")
        return {"success": False, "error": politician_result.get('error')}
    
    print(f"Successfully built politician latent space with {politician_result['n_components']} dimensions")
    
    # Set politician party affiliations
    # This is now redundant as the affiliations are already set in build_politician_latent_space
    # multi_space.set_politician_party_affiliations(politician_party_mapping)
    
    # Analyze results
    results = {
        "success": True,
        "party_count": len(party_texts),
        "politician_count": len(processed_politician_texts),
        "analysis": {}
    }
    
    # Compare politicians to their parties
    print("\n=== Comparing Politicians to Their Parties ===")
    politician_comparisons = {}
    
    for politician in multi_space.politician_latent_space.corpus_labels:
        comparison = multi_space.compare_politician_to_party(politician)
        
        if "error" not in comparison:
            print(f"\n{politician} compared to {comparison['party']}:")
            print(f"  Similarity: {comparison['similarity']:.4f}")
            print("  Differences on expert axes:")
            
            for axis, diff in comparison["expert_axes_diff"].items():
                direction = "more" if diff > 0 else "less"
                print(f"    {axis}: {abs(diff):.4f} ({direction})")
                
            politician_comparisons[politician] = {
                "party": comparison["party"],
                "similarity": float(comparison["similarity"]),
                "expert_axes_diff": {
                    axis: float(diff) for axis, diff in comparison["expert_axes_diff"].items()
                }
            }
        else:
            print(f"\nError comparing {politician}: {comparison['error']}")
    
    results["analysis"]["politician_party_comparisons"] = politician_comparisons
    
    # Find party outliers
    print("\n=== Finding Party Outliers ===")
    party_outliers = {}
    
    for party in multi_space.party_latent_space.corpus_labels:
        outliers = multi_space.find_party_outliers(party, threshold=0.2)
        
        if "error" not in outliers:
            print(f"\n{party} outliers ({outliers['outlier_count']}):")
            
            party_outliers[party] = {
                "count": outliers['outlier_count'],
                "outliers": []
            }
            
            for outlier in outliers["outliers"]:
                print(f"  {outlier['politician']}: similarity = {outlier['similarity']:.4f}")
                
                for axis, diff in outlier["expert_axes_diff"].items():
                    direction = "more" if diff > 0 else "less"
                    print(f"    {axis}: {abs(diff):.4f} ({direction})")
                
                party_outliers[party]["outliers"].append({
                    "politician": outlier["politician"],
                    "similarity": float(outlier["similarity"]),
                    "expert_axes_diff": {
                        axis: float(diff) for axis, diff in outlier["expert_axes_diff"].items()
                    }
                })
        else:
            print(f"\nError finding outliers for {party}: {outliers['error']}")
    
    results["analysis"]["party_outliers"] = party_outliers
    
    # Calculate party cohesion
    print("\n=== Party Cohesion ===")
    cohesion_results = {}
    
    for party in multi_space.party_latent_space.corpus_labels:
        cohesion = multi_space.calculate_party_cohesion(party)
        
        if "error" not in cohesion:
            print(f"\n{party}: {cohesion['cohesion']:.4f}")
            print(f"  {cohesion['interpretation']}")
            print(f"  Politicians: {cohesion['politician_count']}")
            
            cohesion_results[party] = {
                "cohesion": float(cohesion["cohesion"]),
                "interpretation": cohesion["interpretation"],
                "politician_count": cohesion["politician_count"]
            }
        else:
            print(f"\nError calculating cohesion for {party}: {cohesion['error']}")
    
    results["analysis"]["party_cohesion"] = cohesion_results
    
    # Compare all parties
    print("\n=== Party Comparison ===")
    comparison = multi_space.compare_all_parties()
    
    if "error" not in comparison:
        print("\nMost similar parties:")
        party1, party2 = comparison["most_similar"]["pair"]
        similarity = comparison["most_similar"]["similarity"]
        print(f"  {party1} and {party2}: {similarity:.4f}")
        
        print("\nMost different parties:")
        party1, party2 = comparison["most_different"]["pair"]
        similarity = comparison["most_different"]["similarity"]
        print(f"  {party1} and {party2}: {similarity:.4f}")
        
        print("\nParty similarity matrix:")
        for party1 in multi_space.party_latent_space.corpus_labels:
            for party2 in multi_space.party_latent_space.corpus_labels:
                if party1 != party2:
                    sim = comparison["comparison_matrix"][party1][party2]
                    print(f"  {party1} vs {party2}: {sim:.4f}")
        
        results["analysis"]["party_comparison"] = {
            "most_similar": {
                "parties": comparison["most_similar"]["pair"],
                "similarity": float(comparison["most_similar"]["similarity"])
            },
            "most_different": {
                "parties": comparison["most_different"]["pair"],
                "similarity": float(comparison["most_different"]["similarity"])
            },
            "similarity_matrix": {
                party1: {
                    party2: float(sim) 
                    for party2, sim in sims.items()
                }
                for party1, sims in comparison["comparison_matrix"].items()
            }
        }
    else:
        print(f"\nError comparing parties: {comparison['error']}")
    
    # Add metadata
    results["metadata"] = {
        "generated_at": datetime.now().isoformat(),
        "model": config['embedding']['model_name'],
        "analysis_type": "multi_level"
    }
    
    print("\n=== Multi-Level Analysis Complete ===")
    return results

if __name__ == '__main__':
    main()