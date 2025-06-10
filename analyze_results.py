import json
import os

# Load the processed data
file_path = 'src/data/processed/political_latent_space.json'
print(f"Loading data from {file_path}...")
print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 1. Movement Coverage Analysis
print("\n=== MOVEMENT COVERAGE ANALYSIS ===")
print(f"Number of movements analyzed: {len(data['movements'])}")
print("Movements with data:")
for movement_id, movement_data in data['movements'].items():
    key_terms_count = len(movement_data.get('key_terms', []))
    print(f"- {movement_id}: {key_terms_count} key terms")
    
    # Check if position data exists
    if 'position' in movement_data:
        print(f"  Has position data: Yes")
    else:
        print(f"  Has position data: No")
        
    # Check if term analysis exists
    if 'term_analysis' in movement_data:
        contexts_count = len(movement_data['term_analysis'].get('contexts', []))
        print(f"  Context analyses: {contexts_count}")
    else:
        print(f"  Context analyses: 0")

# 2. Key Term Analysis
print("\n=== KEY TERM ANALYSIS ===")
for movement_id, movement_data in data['movements'].items():
    if movement_data.get('key_terms'):
        print(f"\n{movement_id} top 10 key terms:")
        for i, (term, score) in enumerate(movement_data.get('key_terms', [])[:10]):
            print(f"  {i+1}. {term} (score: {score:.4f})")

# 3. Context Window Analysis
print("\n=== CONTEXT WINDOW ANALYSIS ===")
for movement_id, movement_data in data['movements'].items():
    if 'term_analyses' not in movement_data:
        print(f"\n{movement_id}: No term analyses found")
        continue
    
    # Get the first term with contexts
    term_analyses = movement_data['term_analyses']
    if not term_analyses:
        print(f"\n{movement_id}: Empty term analyses dictionary")
        continue
    
    # Print information about all analyzed terms
    print(f"\n{movement_id} term analyses:")
    for term, analysis in term_analyses.items():
        occurrences = analysis.get('occurrences', 0)
        contexts_count = len(analysis.get('contexts', []))
        print(f"  - {term}: {occurrences} occurrences, {contexts_count} contexts")
    
    # Show a sample context from the first term
    first_term = next(iter(term_analyses))
    first_analysis = term_analyses[first_term]
    contexts = first_analysis.get('contexts', [])
    
    if contexts:
        print(f"\n{movement_id} sample context for term '{first_term}':")
        context = contexts[0]
        context_text = context.get('text', '')[:200] + "..." if len(context.get('text', '')) > 200 else context.get('text', '')
        print(f"  Context: {context_text}")
        print(f"  Window size: {len(context_text)} chars")
        
        # Check if there are semantic clusters
        if 'semantic_clusters' in first_analysis and first_analysis['semantic_clusters']:
            print(f"  Has semantic clusters: Yes")
        else:
            print(f"  Has semantic clusters: No")

# 4. Metadata Analysis
print("\n=== METADATA ANALYSIS ===")
if 'metadata' in data:
    metadata = data['metadata']
    print("Metadata keys:", list(metadata.keys()))
    
    if 'generated_at' in metadata:
        print(f"Generated at: {metadata['generated_at']}")
    
    if 'model' in metadata:
        print(f"Model: {metadata['model']}")

# 5. Term Analysis Structure
print("\n=== TERM ANALYSIS STRUCTURE ===")
for movement_id, movement_data in data['movements'].items():
    if 'term_analyses' in movement_data and movement_data['term_analyses']:
        first_term = next(iter(movement_data['term_analyses']))
        analysis = movement_data['term_analyses'][first_term]
        print(f"\n{movement_id} term analysis structure for '{first_term}':")
        print(f"  Keys in analysis: {list(analysis.keys())}")
        
        if 'contexts' in analysis and analysis['contexts']:
            first_context = analysis['contexts'][0]
            print(f"  Keys in context: {list(first_context.keys())}")
            
            # Check for embedding in context
            if 'embedding' in first_context:
                print(f"  Context has embedding: Yes (length: {len(first_context['embedding'])})")

# 5. Expert Dimension Analysis
print("\n=== EXPERT DIMENSION ANALYSIS ===")
for movement_id, movement_data in data['movements'].items():
    if 'position' in movement_data and 'expert_dimensions' in movement_data['position']:
        print(f"\n{movement_id} position in expert dimensions:")
        
        # Print axes positions
        if 'axes' in movement_data['position']['expert_dimensions']:
            axes = movement_data['position']['expert_dimensions']['axes']
            print("  Axes:")
            for axis_name, position in axes.items():
                print(f"    {axis_name}: {position:.4f}")
        
        # Print raw similarities
        if 'raw_similarities' in movement_data['position']['expert_dimensions']:
            similarities = movement_data['position']['expert_dimensions']['raw_similarities']
            print("  Raw similarities:")
            for anchor, similarity in similarities.items():
                print(f"    {anchor}: {similarity:.4f}")

print("\nAnalysis complete!")
