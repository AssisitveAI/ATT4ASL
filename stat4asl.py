"""
Main ATT4ASL class implementation.

This module provides the main entry point for the ATT4ASL library.
"""

import os
from typing import List, Dict, Any, Optional, Tuple, Union

from .models import ProcessedText, AdaptedText
from .preprocessor.preprocessor import Preprocessor
from .replacement.asl_lex_dictionary import ASLLexDictionary
from .replacement.similarity import SimilarityCalculator
from .replacement.replacement_engine import ReplacementEngine
from .postprocessor.postprocessor import Postprocessor
from .evaluation.evaluator import MetricsCalculator, Evaluator


class ATT4ASL:
    """
    Main class for the ATT4ASL library.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ATT4ASL library with optional configuration.
        
        Args:
            config: Dictionary containing configuration options. If None, default configuration is used.
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            # ASL-LEX dictionary
            "asl_lex_path": self.config.get("asl_lex_path", None),
            
            # Output formatting
            "output_format": "replacement_only",  # or "original_with_replacement"
            
            # Preprocessing
            "tokenizer": "spacy",  # or "nltk"
            "lemmatizer": "spacy",  # or "nltk"
            "lowercase": True,
            
            # Replacement engine
            "embedding_model": "fasttext",  # or "word2vec", "glove"
            "embedding_path": None,
            "use_wordnet": True,
            "use_levenshtein": True,
            "min_similarity_threshold": 0.6,
            "pos_match_weight": 0.3,
            "semantic_similarity_weight": 0.7,
            "levenshtein_weight": 0.1,
            
            # Evaluation
            "metrics": ["tar", "avc", "semantic_similarity", "fk_original", "fk_adapted", "fk_reduction"]
        }
        
        # Update default config with provided config
        for key, value in self.config.items():
            self.default_config[key] = value
        
        self.config = self.default_config
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components of the library."""
        # Check for ASL-LEX path
        asl_lex_path = self.config.get("asl_lex_path")
        if not asl_lex_path:
            raise ValueError("ASL-LEX path must be provided in configuration")
        
        # Initialize ASL-LEX dictionary
        self.asl_lex_dict = ASLLexDictionary(asl_lex_path)
        
        # Initialize preprocessor
        self.preprocessor = Preprocessor(self.config)
        
        # Initialize similarity calculator
        self.similarity_calc = SimilarityCalculator(self.config)
        
        # Initialize replacement engine
        self.replacement_engine = ReplacementEngine(
            self.asl_lex_dict, self.similarity_calc, self.config
        )
        
        # Initialize postprocessor
        self.postprocessor = Postprocessor(self.config)
        
        # Initialize metrics calculator
        self.metrics_calc = MetricsCalculator(self.asl_lex_dict.get_all_headwords())
        
        # Initialize evaluator
        self.evaluator = Evaluator(self.metrics_calc, self.config)
    
    def adapt_text(self, text: str, format_type: Optional[str] = None) -> str:
        """
        Adapt text for ASL users by replacing non-ASL-LEX words with appropriate ASL-LEX headwords.
        
        Args:
            text: Input text to adapt
            format_type: Output format type ("replacement_only" or "original_with_replacement")
                If None, uses the format specified in configuration
                
        Returns:
            Adapted text as a string
        """
        if not format_type:
            format_type = self.config.get("output_format", "replacement_only")
        
        # Process text
        processed_text = self.preprocessor.process(text)
        
        # Replace non-ASL-LEX words
        adapted_text = self.replacement_engine.process_text(processed_text)
        
        # Format output
        formatted_text = self.postprocessor.format_output(adapted_text, format_type)
        
        return formatted_text
    
    def adapt_file(self, input_path: str, output_path: str, 
                   format_type: Optional[str] = None) -> None:
        """
        Adapt text from a file and save the result to another file.
        
        Args:
            input_path: Path to input file
            output_path: Path to output file
            format_type: Output format type ("replacement_only" or "original_with_replacement")
                If None, uses the format specified in configuration
        """
        if not format_type:
            format_type = self.config.get("output_format", "replacement_only")
        
        # Process file
        processed_text = self.preprocessor.process_file(input_path)
        
        # Replace non-ASL-LEX words
        adapted_text = self.replacement_engine.process_text(processed_text)
        
        # Save to file
        self.postprocessor.save_to_file(adapted_text, output_path, format_type)
    
    def get_detailed_adaptation(self, text: str) -> AdaptedText:
        """
        Get detailed adaptation information including original words, replacements, and scores.
        
        Args:
            text: Input text to adapt
            
        Returns:
            AdaptedText object containing detailed adaptation information
        """
        # Process text
        processed_text = self.preprocessor.process(text)
        
        # Replace non-ASL-LEX words
        adapted_text = self.replacement_engine.process_text(processed_text)
        
        return adapted_text
    
    def evaluate(self, original_text: str, adapted_text: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate adaptation quality using multiple metrics.
        
        Args:
            original_text: Original text
            adapted_text: Adapted text (if None, original_text will be adapted first)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if adapted_text is None:
            # Adapt the text first
            detailed_adaptation = self.get_detailed_adaptation(original_text)
        else:
            # Process both texts to get detailed information
            processed_original = self.preprocessor.process(original_text)
            processed_adapted = self.preprocessor.process(adapted_text)
            
            # Create a synthetic AdaptedText object
            from .models import AdaptedSentence, AdaptedToken
            
            adapted_sentences = []
            for i, (orig_sent, adapt_sent) in enumerate(zip(processed_original.sentences, processed_adapted.sentences)):
                adapted_tokens = []
                
                for j, (orig_token, adapt_token) in enumerate(zip(orig_sent.tokens, adapt_sent.tokens)):
                    was_replaced = orig_token.text.lower() != adapt_token.text.lower()
                    
                    adapted_tokens.append(AdaptedToken(
                        original_text=orig_token.text,
                        adapted_text=adapt_token.text,
                        similarity_score=0.8 if was_replaced else 1.0,  # Placeholder
                        replacement_source="manual" if was_replaced else "unchanged"
                    ))
                
                adapted_sentences.append(AdaptedSentence(
                    original_tokens=orig_sent.tokens,
                    adapted_tokens=adapted_tokens,
                    original_text=orig_sent.original_text,
                    adapted_text=adapt_sent.original_text
                ))
            
            detailed_adaptation = AdaptedText(
                original_text=original_text,
                adapted_sentences=adapted_sentences,
                stats={}
            )
        
        # Evaluate
        metrics = self.evaluator.evaluate(detailed_adaptation)
        
        return metrics
    
    def evaluate_file(self, input_path: str, output_path: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate adaptation quality for a file.
        
        Args:
            input_path: Path to original text file
            output_path: Path to adapted text file (if None, input will be adapted first)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Read input file
        with open(input_path, 'r', encoding='utf-8') as f:
            original_text = f.read()
        
        if output_path and os.path.exists(output_path):
            # Read adapted text from file
            with open(output_path, 'r', encoding='utf-8') as f:
                adapted_text = f.read()
            
            return self.evaluate(original_text, adapted_text)
        else:
            # Adapt the text first
            return self.evaluate(original_text)
    
    def evaluate_corpus(self, corpus_path: str, output_dir: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate adaptation quality on a corpus of texts.
        
        Args:
            corpus_path: Path to directory containing corpus files
            output_dir: Path to directory for saving adapted files (if None, no files are saved)
            
        Returns:
            Dictionary mapping file names to evaluation metrics
        """
        if not os.path.isdir(corpus_path):
            raise ValueError(f"Corpus path is not a directory: {corpus_path}")
        
        results = {}
        adapted_texts = []
        
        # Process each file in the corpus
        for filename in os.listdir(corpus_path):
            if not filename.endswith('.txt'):
                continue
            
            input_path = os.path.join(corpus_path, filename)
            
            # Check if there's a corresponding adapted file
            adapted_file = None
            if output_dir and os.path.isdir(output_dir):
                adapted_file = os.path.join(output_dir, filename)
                if not os.path.exists(adapted_file):
                    adapted_file = None
            
            # Evaluate
            metrics = self.evaluate_file(input_path, adapted_file)
            results[filename] = metrics
            
            # Adapt and save if output_dir is provided
            if output_dir and not adapted_file:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, filename)
                self.adapt_file(input_path, output_path)
        
        return results
    
    def is_asl_lex_word(self, word: str) -> bool:
        """
        Check if a word is in the ASL-LEX dictionary.
        
        Args:
            word: Word to check
            
        Returns:
            True if the word is in ASL-LEX, False otherwise
        """
        return self.asl_lex_dict.is_asl_lex_word(word)
    
    def find_similar_asl_lex_words(self, word: str, pos: Optional[str] = None, 
                                  top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Find similar ASL-LEX words for a given word.
        
        Args:
            word: Word to find similar words for
            pos: Part of speech (optional)
            top_n: Number of similar words to return
            
        Returns:
            List of (word, similarity_score) tuples
        """
        # Get candidate headwords with the same POS
        if pos:
            candidates = self.asl_lex_dict.get_headwords_by_pos(pos)
        else:
            candidates = self.asl_lex_dict.get_all_headwords()
        
        # Find similar words
        similar_words = self.similarity_calc.find_most_similar(word, candidates, pos, top_n=top_n)
        
        # Format results
        return [(word, score) for word, score, _ in similar_words]
    
    def get_asl_lex_words(self, pos: Optional[str] = None) -> List[str]:
        """
        Get all ASL-LEX words, optionally filtered by part of speech.
        
        Args:
            pos: Part of speech filter (optional)
            
        Returns:
            List of ASL-LEX words
        """
        if pos:
            return self.asl_lex_dict.get_headwords_by_pos(pos)
        else:
            return self.asl_lex_dict.get_all_headwords()
    
    def get_configuration(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Dictionary of current configuration options
        """
        return self.config.copy()
    
    def set_configuration(self, config: Dict[str, Any]) -> None:
        """
        Update configuration.
        
        Args:
            config: Dictionary of configuration options to update
        """
        # Update configuration
        for key, value in config.items():
            self.config[key] = value
        
        # Reinitialize components if necessary
        if any(key in config for key in [
            "asl_lex_path", "tokenizer", "lemmatizer", "embedding_model", 
            "embedding_path", "use_wordnet", "use_levenshtein"
        ]):
            self._initialize_components()
