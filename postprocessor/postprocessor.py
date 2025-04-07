"""
Postprocessor module for the ATT4ASL library.

This module handles formatting and finalizing the adapted text.
"""

import os
from typing import List, Dict, Any, Optional, Tuple

from ..models import AdaptedText


class Postprocessor:
    """
    Postprocessor for formatting and finalizing adapted text.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the postprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    def format_output(self, adapted_text: AdaptedText, 
                      format_type: str = "replacement_only") -> str:
        """
        Format the adapted text according to specified format type.
        
        Args:
            adapted_text: Adapted text
            format_type: Format type ("replacement_only" or "original_with_replacement")
            
        Returns:
            Formatted text
        """
        if format_type == "replacement_only":
            return adapted_text.adapted_text
        
        elif format_type == "original_with_replacement":
            formatted_sentences = []
            
            for sentence in adapted_text.adapted_sentences:
                formatted_tokens = []
                
                for token in sentence.adapted_tokens:
                    if token.was_replaced:
                        # Format as "replacement (original)"
                        formatted_tokens.append(f"{token.adapted_text} ({token.original_text})")
                    else:
                        formatted_tokens.append(token.adapted_text)
                
                formatted_sentences.append(" ".join(formatted_tokens))
            
            return "\n".join(formatted_sentences)
        
        else:
            raise ValueError(f"Unknown format type: {format_type}")
    
    def save_to_file(self, adapted_text: AdaptedText, 
                     file_path: str, format_type: str = "replacement_only") -> None:
        """
        Save the adapted text to a file.
        
        Args:
            adapted_text: Adapted text
            file_path: Output file path
            format_type: Format type ("replacement_only" or "original_with_replacement")
        """
        formatted_text = self.format_output(adapted_text, format_type)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(formatted_text)
    
    def save_report(self, adapted_text: AdaptedText, file_path: str) -> None:
        """
        Save a detailed report of the adaptation.
        
        Args:
            adapted_text: Adapted text
            file_path: Output file path
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Determine file extension
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.html':
            # Save as HTML
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(adapted_text.to_html())
        
        elif ext.lower() == '.json':
            # Save as JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(adapted_text.to_json())
        
        else:
            # Save as text report
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("ATT4ASL Adaptation Report\n")
                f.write("=========================\n\n")
                
                # Statistics
                f.write("Statistics:\n")
                for key, value in adapted_text.stats.items():
                    if isinstance(value, float):
                        f.write(f"- {key}: {value:.2f}\n")
                    else:
                        f.write(f"- {key}: {value}\n")
                f.write("\n")
                
                # Original text
                f.write("Original Text:\n")
                f.write("--------------\n")
                f.write(adapted_text.original_text)
                f.write("\n\n")
                
                # Adapted text
                f.write("Adapted Text:\n")
                f.write("-------------\n")
                f.write(adapted_text.adapted_text)
                f.write("\n\n")
                
                # Replacements
                f.write("Replacements:\n")
                f.write("-------------\n")
                
                for sentence in adapted_text.adapted_sentences:
                    for token in sentence.adapted_tokens:
                        if token.was_replaced:
                            f.write(f"- '{token.original_text}' -> '{token.adapted_text}' ")
                            f.write(f"(similarity: {token.similarity_score:.2f}, ")
                            f.write(f"source: {token.replacement_source})\n")
