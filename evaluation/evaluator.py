"""
Evaluation module for the ATT4ASL library.

This module provides metrics and evaluation tools for assessing the quality of adaptations.
"""

import os
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter

from ..models import AdaptedText


class MetricsCalculator:
    """
    Calculator for various evaluation metrics.
    """
    
    def __init__(self, asl_lex_words: List[str]):
        """
        Initialize the metrics calculator.
        
        Args:
            asl_lex_words: List of ASL-LEX headwords
        """
        self.asl_lex_words = set(word.lower() for word in asl_lex_words)
    
    def calculate_tar(self, adapted_text: AdaptedText) -> float:
        """
        Calculate Text Adaptation Rate (TAR).
        
        Args:
            adapted_text: Adapted text
            
        Returns:
            TAR value between 0 and 1
        """
        # TAR is already calculated in the replacement engine
        return adapted_text.stats.get('replacement_rate', 0.0)
    
    def calculate_avc(self, adapted_text: AdaptedText) -> float:
        """
        Calculate ASL Vocabulary Coverage (AVC).
        
        Args:
            adapted_text: Adapted text
            
        Returns:
            AVC value between 0 and 1
        """
        total_tokens = 0
        asl_lex_tokens = 0
        
        for sentence in adapted_text.adapted_sentences:
            for token in sentence.adapted_tokens:
                # Skip very short words and punctuation
                if len(token.adapted_text) <= 2 or all(c in '.,;:!?-()[]{}"\'' for c in token.adapted_text):
                    continue
                
                total_tokens += 1
                if token.adapted_text.lower() in self.asl_lex_words:
                    asl_lex_tokens += 1
        
        return asl_lex_tokens / total_tokens if total_tokens > 0 else 0.0
    
    def calculate_flesch_kincaid(self, text: str) -> float:
        """
        Calculate Flesch-Kincaid Grade Level.
        
        Args:
            text: Input text
            
        Returns:
            Flesch-Kincaid Grade Level score
        """
        # Count sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        num_sentences = len(sentences)
        
        if num_sentences == 0:
            return 0.0
        
        # Count words
        words = re.findall(r'\b\w+\b', text.lower())
        num_words = len(words)
        
        if num_words == 0:
            return 0.0
        
        # Count syllables (approximate)
        syllable_count = 0
        for word in words:
            syllable_count += self._count_syllables(word)
        
        # Calculate Flesch-Kincaid Grade Level
        fk_grade = 0.39 * (num_words / num_sentences) + 11.8 * (syllable_count / num_words) - 15.59
        
        return max(0.0, fk_grade)  # Ensure non-negative
    
    def _count_syllables(self, word: str) -> int:
        """
        Count syllables in a word (approximate).
        
        Args:
            word: Input word
            
        Returns:
            Number of syllables
        """
        # Remove non-alphabetic characters
        word = re.sub(r'[^a-z]', '', word.lower())
        
        if not word:
            return 0
        
        # Count vowel groups
        vowels = 'aeiouy'
        count = 0
        prev_is_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
        
        # Adjust for silent 'e' at the end
        if word.endswith('e') and len(word) > 2 and word[-2] not in vowels:
            count -= 1
        
        # Ensure at least one syllable
        return max(1, count)
    
    def calculate_semantic_similarity(self, original_text: str, adapted_text: str) -> float:
        """
        Calculate semantic similarity between original and adapted text.
        
        Args:
            original_text: Original text
            adapted_text: Adapted text
            
        Returns:
            Semantic similarity score between 0 and 1
        """
        # This is a simplified implementation
        # For a real implementation, you would use sentence embeddings
        
        # Count word overlap
        original_words = set(re.findall(r'\b\w+\b', original_text.lower()))
        adapted_words = set(re.findall(r'\b\w+\b', adapted_text.lower()))
        
        if not original_words or not adapted_words:
            return 0.0
        
        # Jaccard similarity
        intersection = len(original_words.intersection(adapted_words))
        union = len(original_words.union(adapted_words))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_all_metrics(self, adapted_text: AdaptedText) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.
        
        Args:
            adapted_text: Adapted text
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Text Adaptation Rate
        metrics['tar'] = self.calculate_tar(adapted_text)
        
        # ASL Vocabulary Coverage
        metrics['avc'] = self.calculate_avc(adapted_text)
        
        # Flesch-Kincaid Grade Level
        metrics['fk_original'] = self.calculate_flesch_kincaid(adapted_text.original_text)
        metrics['fk_adapted'] = self.calculate_flesch_kincaid(adapted_text.adapted_text)
        metrics['fk_reduction'] = max(0, metrics['fk_original'] - metrics['fk_adapted'])
        
        # Semantic Similarity
        metrics['semantic_similarity'] = self.calculate_semantic_similarity(
            adapted_text.original_text, adapted_text.adapted_text
        )
        
        return metrics


class Evaluator:
    """
    Evaluator for assessing adaptation quality.
    """
    
    def __init__(self, metrics_calculator: MetricsCalculator, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluator.
        
        Args:
            metrics_calculator: Metrics calculator
            config: Configuration dictionary
        """
        self.metrics_calc = metrics_calculator
        self.config = config or {}
        self.metrics = self.config.get('metrics', [
            'tar', 'avc', 'fk_original', 'fk_adapted', 'fk_reduction', 'semantic_similarity'
        ])
    
    def evaluate(self, adapted_text: AdaptedText) -> Dict[str, float]:
        """
        Evaluate adaptation quality using multiple metrics.
        
        Args:
            adapted_text: Adapted text
            
        Returns:
            Dictionary of evaluation metrics
        """
        all_metrics = self.metrics_calc.calculate_all_metrics(adapted_text)
        
        # Filter metrics based on configuration
        if self.metrics:
            return {k: v for k, v in all_metrics.items() if k in self.metrics}
        
        return all_metrics
    
    def evaluate_corpus(self, adapted_texts: List[AdaptedText]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate adaptation quality on a corpus of texts.
        
        Args:
            adapted_texts: List of adapted texts
            
        Returns:
            Dictionary mapping text IDs to evaluation metrics
        """
        results = {}
        
        for i, adapted_text in enumerate(adapted_texts):
            text_id = adapted_text.metadata.get('id', f'text_{i}')
            results[text_id] = self.evaluate(adapted_text)
        
        return results
    
    def calculate_average_metrics(self, corpus_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate average metrics across a corpus.
        
        Args:
            corpus_results: Results from evaluate_corpus
            
        Returns:
            Dictionary of average metric values
        """
        if not corpus_results:
            return {}
        
        # Initialize with first result's keys
        first_result = next(iter(corpus_results.values()))
        avg_metrics = {k: 0.0 for k in first_result.keys()}
        
        # Sum all values
        for result in corpus_results.values():
            for k, v in result.items():
                avg_metrics[k] += v
        
        # Calculate averages
        for k in avg_metrics:
            avg_metrics[k] /= len(corpus_results)
        
        return avg_metrics
    
    def generate_report(self, corpus_results: Dict[str, Dict[str, float]], 
                        output_path: str) -> None:
        """
        Generate an evaluation report.
        
        Args:
            corpus_results: Results from evaluate_corpus
            output_path: Path to save the report
        """
        avg_metrics = self.calculate_average_metrics(corpus_results)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("ATT4ASL Evaluation Report\n")
            f.write("=========================\n\n")
            
            # Average metrics
            f.write("Average Metrics:\n")
            f.write("----------------\n")
            for k, v in avg_metrics.items():
                f.write(f"{k}: {v:.4f}\n")
            f.write("\n")
            
            # Individual text metrics
            f.write("Individual Text Metrics:\n")
            f.write("-----------------------\n")
            for text_id, metrics in corpus_results.items():
                f.write(f"\n{text_id}:\n")
                for k, v in metrics.items():
                    f.write(f"  {k}: {v:.4f}\n")
