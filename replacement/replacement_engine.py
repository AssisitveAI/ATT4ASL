"""
Replacement engine for the ATT4ASL library.

This module handles the core functionality of replacing non-ASL-LEX words
with appropriate ASL-LEX headwords.
"""

import os
from typing import List, Dict, Any, Optional, Tuple, Set

from ..models import Token, Sentence, ProcessedText, AdaptedToken, AdaptedSentence, AdaptedText
from .asl_lex_dictionary import ASLLexDictionary
from .similarity import SimilarityCalculator


class ReplacementEngine:
    """
    Core engine for replacing non-ASL-LEX words with appropriate ASL-LEX headwords.
    """
    
    def __init__(self, asl_lex_dictionary: ASLLexDictionary, 
                 similarity_calculator: SimilarityCalculator,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the replacement engine.
        
        Args:
            asl_lex_dictionary: ASL-LEX dictionary
            similarity_calculator: Similarity calculator
            config: Configuration dictionary
        """
        self.asl_lex_dict = asl_lex_dictionary
        self.similarity_calc = similarity_calculator
        self.config = config or {}
        
        # Configuration
        self.min_similarity_threshold = self.config.get('min_similarity_threshold', 0.6)
        self.max_candidates = self.config.get('max_candidates', 10)
    
    def find_replacement(self, word: str, pos: str, 
                         context: Optional[List[str]] = None) -> Tuple[Optional[str], float, str]:
        """
        Find the best ASL-LEX replacement for a word.
        
        Args:
            word: Word to replace
            pos: Part of speech
            context: Context words (optional)
            
        Returns:
            Tuple of (replacement word, similarity score, replacement source)
        """
        # Check if word is already in ASL-LEX
        if self.asl_lex_dict.is_asl_lex_word_with_pos(word, pos):
            return word, 1.0, "unchanged"
        
        # Map POS to ASL-LEX POS
        asl_lex_pos = self.asl_lex_dict.map_pos_to_asl_lex(pos)
        
        # Get candidate headwords with the same POS
        candidates = self.asl_lex_dict.get_headwords_by_pos(asl_lex_pos)
        
        # If no candidates with matching POS, use all headwords
        if not candidates:
            candidates = self.asl_lex_dict.get_all_headwords()
        
        # Limit number of candidates for efficiency
        if len(candidates) > self.max_candidates:
            # Try to find synonyms first using WordNet
            synonyms = self.similarity_calc.wordnet_interface.get_synonyms(word, pos)
            asl_lex_synonyms = [s for s in synonyms if self.asl_lex_dict.is_asl_lex_word(s)]
            
            if asl_lex_synonyms:
                # Found synonyms in ASL-LEX
                best_synonym = asl_lex_synonyms[0]
                return best_synonym, 0.9, "wordnet"
            
            # If no synonyms found, use embedding model to find similar words
            if hasattr(self.similarity_calc, 'embedding_model'):
                similar_words = self.similarity_calc.embedding_model.find_most_similar(
                    word, candidates, top_n=self.max_candidates
                )
                candidates = [w for w, _ in similar_words]
        
        # If still no candidates, return None
        if not candidates:
            return None, 0.0, "none"
        
        # Find most similar candidates
        similar_candidates = self.similarity_calc.find_most_similar(
            word, candidates, pos, top_n=5
        )
        
        # Get the best candidate
        if similar_candidates:
            best_candidate, similarity, scores = similar_candidates[0]
            
            # Check if similarity is above threshold
            if similarity >= self.min_similarity_threshold:
                # Determine replacement source
                if scores.get('wordnet', 0) > scores.get('embedding', 0) and scores.get('wordnet', 0) > 0.5:
                    source = "wordnet"
                elif scores.get('embedding', 0) > 0.5:
                    source = "embedding"
                elif scores.get('levenshtein', 0) > 0.8:
                    source = "levenshtein"
                else:
                    source = "combined"
                
                return best_candidate, similarity, source
        
        # No suitable replacement found
        return None, 0.0, "none"
    
    def process_text(self, processed_text: ProcessedText) -> AdaptedText:
        """
        Process entire text and replace non-ASL-LEX words.
        
        Args:
            processed_text: Processed text
            
        Returns:
            Adapted text
        """
        adapted_sentences = []
        total_tokens = 0
        replaced_tokens = 0
        
        for sentence in processed_text.sentences:
            original_tokens = sentence.tokens
            adapted_tokens = []
            
            for token in original_tokens:
                # Skip very short words (likely articles, prepositions, etc.)
                if len(token.text) <= 2:
                    adapted_tokens.append(AdaptedToken(
                        original_text=token.text,
                        adapted_text=token.text
                    ))
                    total_tokens += 1
                    continue
                
                # Check if token is already in ASL-LEX
                token.is_asl_lex = self.asl_lex_dict.is_asl_lex_word_with_pos(token.text, token.pos)
                
                if token.is_asl_lex:
                    # No need to replace
                    adapted_tokens.append(AdaptedToken(
                        original_text=token.text,
                        adapted_text=token.text
                    ))
                else:
                    # Find replacement
                    replacement, similarity, source = self.find_replacement(
                        token.text, token.pos
                    )
                    
                    if replacement and replacement != token.text:
                        adapted_tokens.append(AdaptedToken(
                            original_text=token.text,
                            adapted_text=replacement,
                            similarity_score=similarity,
                            replacement_source=source
                        ))
                        replaced_tokens += 1
                    else:
                        # No suitable replacement found, keep original
                        adapted_tokens.append(AdaptedToken(
                            original_text=token.text,
                            adapted_text=token.text
                        ))
                
                total_tokens += 1
            
            # Reconstruct adapted sentence text
            adapted_text = ' '.join([token.adapted_text for token in adapted_tokens])
            
            adapted_sentences.append(AdaptedSentence(
                original_tokens=original_tokens,
                adapted_tokens=adapted_tokens,
                original_text=sentence.original_text,
                adapted_text=adapted_text
            ))
        
        # Calculate statistics
        replacement_rate = replaced_tokens / total_tokens if total_tokens > 0 else 0
        
        # Create adapted text
        adapted_text = AdaptedText(
            original_text='\n'.join([s.original_text for s in processed_text.sentences]),
            adapted_sentences=adapted_sentences,
            metadata=processed_text.metadata,
            stats={
                'total_tokens': total_tokens,
                'replaced_tokens': replaced_tokens,
                'replacement_rate': replacement_rate
            }
        )
        
        return adapted_text
