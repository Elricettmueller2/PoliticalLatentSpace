"""
Data exporting functionality for political latent space analysis.
"""

import os
import json
import numpy as np
from typing import Dict, Any, Optional, Union
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles NumPy types.
    """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class DataExporter:
    """
    Exports analysis results to various formats.
    """
    
    def __init__(self, output_dir='src/data/processed'):
        """
        Initialize the data exporter.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        self._ensure_output_dir()
    
    def export_to_json(self, data: Dict[str, Any], filename: str) -> str:
        """
        Export data to a JSON file.
        
        Args:
            data: Data to export
            filename: Output filename
            
        Returns:
            Path to the output file
        """
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            print(f"Data exported to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return ""
    
    def export_to_csv(self, data: Dict[str, Any], filename: str) -> str:
        """
        Export data to CSV format.
        
        Args:
            data: Data to export
            filename: Output filename
            
        Returns:
            Path to the output file
        """
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            # Convert to DataFrame
            if "movements" in data:
                # Extract positions for each movement
                positions = {}
                for movement, info in data["movements"].items():
                    if "position" in info and "expert_dimensions" in info["position"]:
                        axes = info["position"]["expert_dimensions"].get("axes", {})
                        positions[movement] = axes
                
                if positions:
                    df = pd.DataFrame.from_dict(positions, orient='index')
                    df.to_csv(output_path)
                    print(f"Positions exported to {output_path}")
                    return output_path
            
            print("Data structure not suitable for CSV export")
            return ""
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return ""
    
    def export_visualization_data(self, data: Dict[str, Any], filename: str) -> str:
        """
        Export data specifically formatted for visualization.
        
        Args:
            data: Analysis data
            filename: Output filename
            
        Returns:
            Path to the output file
        """
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            # Extract visualization-specific data
            viz_data = {
                "movements": {},
                "dimensions": [],
                "metadata": data.get("metadata", {})
            }
            
            # Process movement data
            if "movements" in data:
                for movement, info in data["movements"].items():
                    if "position" in info:
                        position = info["position"]
                        
                        # Extract expert dimensions
                        expert_dims = {}
                        if "expert_dimensions" in position and "axes" in position["expert_dimensions"]:
                            expert_dims = position["expert_dimensions"]["axes"]
                        
                        # Extract learned dimensions
                        learned_dims = {}
                        if "learned_dimensions" in position:
                            learned_dims = position["learned_dimensions"]
                        
                        # Combine dimensions
                        all_dims = {**expert_dims, **learned_dims}
                        
                        # Extract key terms
                        key_terms = info.get("key_terms", [])
                        term_list = [term for term, _ in key_terms[:10]] if key_terms else []
                        
                        viz_data["movements"][movement] = {
                            "position": all_dims,
                            "key_terms": term_list
                        }
            
            # Extract dimension information
            dimensions = set()
            for movement_data in viz_data["movements"].values():
                dimensions.update(movement_data["position"].keys())
            
            viz_data["dimensions"] = sorted(list(dimensions))
            
            # Export to JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(viz_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            
            print(f"Visualization data exported to {output_path}")
            return output_path
        
        except Exception as e:
            print(f"Error exporting visualization data: {e}")
            return ""
    
    def export_term_analysis(self, term_data: Dict[str, Any], filename: str) -> str:
        """
        Export term analysis data.
        
        Args:
            term_data: Term analysis data
            filename: Output filename
            
        Returns:
            Path to the output file
        """
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            # Simplify the term analysis data for export
            simplified_data = {}
            
            for term, analysis in term_data.items():
                # Extract basic information
                simplified_analysis = {
                    "occurrences": analysis.get("occurrences", 0),
                    "contexts_count": len(analysis.get("contexts", [])),
                    "window_size": analysis.get("window_size", 0)
                }
                
                # Extract semantic clusters if available
                clusters = analysis.get("semantic_clusters", {})
                if clusters:
                    cluster_sizes = {
                        f"cluster_{cluster_id}": len(contexts) 
                        for cluster_id, contexts in clusters.items()
                    }
                    simplified_analysis["clusters"] = cluster_sizes
                
                simplified_data[term] = simplified_analysis
            
            # Export to JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(simplified_data, f, ensure_ascii=False, indent=2)
            
            print(f"Term analysis exported to {output_path}")
            return output_path
        
        except Exception as e:
            print(f"Error exporting term analysis: {e}")
            return ""
    
    def _ensure_output_dir(self):
        """
        Ensure the output directory exists.
        """
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating output directory: {e}")
