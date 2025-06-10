"""
Data loading functionality for political texts.
"""

import os
import glob
from typing import Dict, List, Any, Optional
import re
import PyPDF2


class PoliticalTextLoader:
    """
    Loads political texts from various sources and formats.
    """
    
    def __init__(self, base_dir='src/data/raw'):
        """
        Initialize the text loader.
        
        Args:
            base_dir: Base directory for raw data
        """
        self.base_dir = base_dir
    
    def load_movement_texts(self, movement_dirs: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Load texts for political movements.
        
        Args:
            movement_dirs: Optional list of movement directories to load.
                          If None, load all directories in base_dir.
            
        Returns:
            Dictionary mapping movement names to concatenated text content
        """
        if movement_dirs is None:
            # Get all subdirectories in base_dir
            try:
                movement_dirs = [d for d in os.listdir(self.base_dir) 
                               if os.path.isdir(os.path.join(self.base_dir, d))]
            except FileNotFoundError:
                print(f"Base directory {self.base_dir} not found.")
                return {}
        
        texts = {}
        for movement in movement_dirs:
            movement_path = os.path.join(self.base_dir, movement)
            if not os.path.isdir(movement_path):
                print(f"Warning: {movement_path} is not a directory. Skipping.")
                continue
            
            print(f"Loading texts for {movement}...")
            movement_text = self.load_movement_directory(movement_path)
            
            if movement_text:
                texts[movement] = movement_text
            else:
                print(f"No text content found for {movement}.")
        
        return texts
    
    def load_movement_directory(self, directory: str) -> str:
        """
        Load all text files from a movement directory and its subdirectories.
        
        Args:
            directory: Path to the movement directory
            
        Returns:
            Concatenated text content
        """
        all_text = ""
        
        # First, load any text files directly in the movement directory
        text_files = glob.glob(os.path.join(directory, "*.txt"))
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        all_text += content + "\n\n"
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        # Also load any PDF files in the directory
        pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
        for pdf_path in pdf_files:
            try:
                pdf_text = self.extract_text_from_pdf(pdf_path)
                if pdf_text.strip():
                    all_text += pdf_text + "\n\n"
            except Exception as e:
                print(f"Error reading PDF {pdf_path}: {e}")
        
        # Then, recursively load from subdirectories
        subdirs = [d for d in os.listdir(directory) 
                  if os.path.isdir(os.path.join(directory, d))]
        
        for subdir in subdirs:
            subdir_path = os.path.join(directory, subdir)
            subdir_text = self.load_movement_directory(subdir_path)
            if subdir_text:
                all_text += subdir_text + "\n\n"
        
        return all_text
    
    def load_specific_file(self, file_path: str) -> str:
        """
        Load a specific text file or PDF file.
        
        Args:
            file_path: Path to the text or PDF file
            
        Returns:
            Text content
        """
        try:
            if file_path.lower().endswith('.pdf'):
                return self.extract_text_from_pdf(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""
    
    def load_movement_file_by_type(self, movement: str, file_type: str) -> str:
        """
        Load a specific type of file for a movement (e.g., manifest, program).
        
        Args:
            movement: Name of the movement
            file_type: Type of file to load (e.g., 'manifest', 'program')
            
        Returns:
            Text content
        """
        movement_path = os.path.join(self.base_dir, movement)
        
        # Look for files matching the type (both txt and pdf)
        txt_pattern = os.path.join(movement_path, f"*{file_type}*.txt")
        pdf_pattern = os.path.join(movement_path, f"*{file_type}*.pdf")
        
        matching_files = glob.glob(txt_pattern) + glob.glob(pdf_pattern)
        
        if not matching_files:
            print(f"No {file_type} file found for {movement}.")
            return ""
        
        # Use the first matching file
        return self.load_specific_file(matching_files[0])
    
    def get_available_movements(self) -> List[str]:
        """
        Get a list of available movement directories.
        
        Returns:
            List of movement names
        """
        try:
            return [d for d in os.listdir(self.base_dir) 
                   if os.path.isdir(os.path.join(self.base_dir, d))]
        except FileNotFoundError:
            print(f"Base directory {self.base_dir} not found.")
            return []
    
    def get_movement_file_structure(self, movement: str) -> Dict[str, Any]:
        """
        Get the file structure for a specific movement.
        
        Args:
            movement: Name of the movement
            
        Returns:
            Dictionary representing the file structure
        """
        movement_path = os.path.join(self.base_dir, movement)
        if not os.path.isdir(movement_path):
            return {"error": f"{movement} directory not found"}
        
        return self._get_directory_structure(movement_path)
    
    def _get_directory_structure(self, path: str) -> Dict[str, Any]:
        """
        Recursively get the structure of a directory.
        
        Args:
            path: Path to the directory
            
        Returns:
            Dictionary representing the directory structure
        """
        structure = {}
        
        try:
            items = os.listdir(path)
            
            # Process files
            files = [f for f in items if os.path.isfile(os.path.join(path, f))]
            if files:
                structure["files"] = files
            
            # Process subdirectories
            subdirs = [d for d in items if os.path.isdir(os.path.join(path, d))]
            if subdirs:
                structure["subdirectories"] = {}
                for subdir in subdirs:
                    subdir_path = os.path.join(path, subdir)
                    structure["subdirectories"][subdir] = self._get_directory_structure(subdir_path)
        
        except Exception as e:
            structure["error"] = str(e)
        
        return structure
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                # Extract text from each page
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                        
            return text
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""
