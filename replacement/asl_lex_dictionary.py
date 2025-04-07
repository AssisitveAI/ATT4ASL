"""
ASL-LEX dictionary module for the ATT4ASL library.

This module handles loading and querying the ASL-LEX dictionary.
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional, Set, Tuple


class ASLLexDictionary:
    """
    ASL-LEX dictionary manager.
    """
    
    def __init__(self, asl_lex_path: str):
        """
        Initialize the ASL-LEX dictionary.
        
        Args:
            asl_lex_path: Path to ASL-LEX CSV file
        """
        if not os.path.exists(asl_lex_path):
            raise FileNotFoundError(f"ASL-LEX file not found: {asl_lex_path}")
        
        self.asl_lex_path = asl_lex_path
        self.data = pd.read_csv(asl_lex_path)
        
        # Create sets for faster lookup
        self.headwords = set(self.data['Headword'].str.lower())
        
        # Create POS-specific sets
        self.pos_headwords = {}
        for pos in self.data['POS'].unique():
            self.pos_headwords[pos] = set(
                self.data[self.data['POS'] == pos]['Headword'].str.lower()
            )
        
        # Map from POS tags in different tagsets to ASL-LEX POS categories
        self.pos_mapping = {
            # NLTK/Penn Treebank to ASL-LEX
            'NN': 'Noun', 'NNS': 'Noun', 'NNP': 'Noun', 'NNPS': 'Noun',
            'VB': 'Verb', 'VBD': 'Verb', 'VBG': 'Verb', 'VBN': 'Verb', 'VBP': 'Verb', 'VBZ': 'Verb',
            'JJ': 'Adjective', 'JJR': 'Adjective', 'JJS': 'Adjective',
            'RB': 'Adverb', 'RBR': 'Adverb', 'RBS': 'Adverb',
            'CD': 'Number',
            
            # spaCy to ASL-LEX
            'NOUN': 'Noun', 'PROPN': 'Noun',
            'VERB': 'Verb',
            'ADJ': 'Adjective',
            'ADV': 'Adverb',
            'NUM': 'Number',
            'ADP': 'Minor', 'CCONJ': 'Minor', 'DET': 'Minor', 'PART': 'Minor', 'PRON': 'Minor',
            'SCONJ': 'Minor', 'INTJ': 'Minor',
        }
    
    def is_asl_lex_word(self, word: str) -> bool:
        """
        Check if a word is in the ASL-LEX dictionary.
        
        Args:
            word: Word to check
            
        Returns:
            True if the word is in ASL-LEX, False otherwise
        """
        return word.lower() in self.headwords
    
    def is_asl_lex_word_with_pos(self, word: str, pos: str) -> bool:
        """
        Check if a word with a specific POS is in the ASL-LEX dictionary.
        
        Args:
            word: Word to check
            pos: Part of speech
            
        Returns:
            True if the word with the given POS is in ASL-LEX, False otherwise
        """
        # Map the POS tag to ASL-LEX POS category
        asl_lex_pos = self.map_pos_to_asl_lex(pos)
        
        if asl_lex_pos in self.pos_headwords:
            return word.lower() in self.pos_headwords[asl_lex_pos]
        return False
    
    def map_pos_to_asl_lex(self, pos: str) -> str:
        """
        Map a POS tag from a different tagset to ASL-LEX POS category.
        
        Args:
            pos: POS tag from another tagset
            
        Returns:
            Corresponding ASL-LEX POS category
        """
        return self.pos_mapping.get(pos, 'Minor')
    
    def get_all_headwords(self) -> List[str]:
        """
        Get all ASL-LEX headwords.
        
        Returns:
            List of all headwords
        """
        return list(self.headwords)
    
    def get_headwords_by_pos(self, pos: str) -> List[str]:
        """
        Get ASL-LEX headwords filtered by part of speech.
        
        Args:
            pos: Part of speech
            
        Returns:
            List of headwords with the given POS
        """
        asl_lex_pos = self.map_pos_to_asl_lex(pos)
        
        if asl_lex_pos in self.pos_headwords:
            return list(self.pos_headwords[asl_lex_pos])
        return []
    
    def get_pos_for_headword(self, word: str) -> Optional[str]:
        """
        Get the part of speech for a headword.
        
        Args:
            word: Headword to look up
            
        Returns:
            Part of speech or None if not found
        """
        word_lower = word.lower()
        if word_lower in self.headwords:
            # Find the row with this headword
            matches = self.data[self.data['Headword'].str.lower() == word_lower]
            if not matches.empty:
                return matches.iloc[0]['POS']
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the ASL-LEX dictionary.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_entries': len(self.data),
            'unique_headwords': len(self.headwords),
            'pos_distribution': {
                pos: len(words) for pos, words in self.pos_headwords.items()
            }
        }
        return stats
