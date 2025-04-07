"""
Preprocessor module for the ATT4ASL library.

This module handles text preprocessing including tokenization, POS tagging, and lemmatization.
"""

import os
from typing import List, Dict, Any, Optional, Union
import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from ..models import Token, Sentence, ProcessedText

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')


class Preprocessor:
    """
    Preprocessor for text analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.tokenizer = self.config.get('tokenizer', 'spacy')
        self.lemmatizer = self.config.get('lemmatizer', 'spacy')
        self.lowercase = self.config.get('lowercase', True)
        
        # Initialize spaCy if needed
        if self.tokenizer == 'spacy' or self.lemmatizer == 'spacy':
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                # Try smaller model if the medium one is not available
                try:
                    self.nlp = spacy.load('en')
                except OSError:
                    raise ValueError("No spaCy English model found. Please install one with: python -m spacy download en_core_web_sm")
        
        # Initialize NLTK lemmatizer if needed
        if self.lemmatizer == 'nltk':
            self.nltk_lemmatizer = WordNetLemmatizer()
    
    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """
        Convert Penn Treebank POS tags to WordNet POS tags.
        
        Args:
            treebank_tag: Penn Treebank POS tag
            
        Returns:
            WordNet POS tag
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # Default to noun
    
    def _process_with_spacy(self, text: str) -> ProcessedText:
        """
        Process text using spaCy.
        
        Args:
            text: Input text
            
        Returns:
            ProcessedText object
        """
        doc = self.nlp(text)
        sentences = []
        
        for sent in doc.sents:
            tokens = []
            for i, token in enumerate(sent):
                # Skip punctuation and whitespace
                if token.is_punct or token.is_space:
                    continue
                
                token_text = token.text.lower() if self.lowercase else token.text
                lemma = token.lemma_.lower() if self.lowercase else token.lemma_
                
                tokens.append(Token(
                    text=token_text,
                    lemma=lemma,
                    pos=token.pos_,
                    is_asl_lex=False,  # Will be updated later
                    index=i
                ))
            
            if tokens:  # Only add sentences with tokens
                sentences.append(Sentence(tokens=tokens, original_text=sent.text))
        
        return ProcessedText(sentences=sentences)
    
    def _process_with_nltk(self, text: str) -> ProcessedText:
        """
        Process text using NLTK.
        
        Args:
            text: Input text
            
        Returns:
            ProcessedText object
        """
        sentences = []
        
        # Tokenize sentences
        sent_texts = sent_tokenize(text)
        
        for sent_text in sent_texts:
            # Tokenize words
            word_tokens = word_tokenize(sent_text)
            
            # POS tagging
            pos_tags = nltk.pos_tag(word_tokens)
            
            tokens = []
            for i, (word, pos) in enumerate(pos_tags):
                # Skip punctuation
                if all(c in '.,;:!?-()[]{}"\'' for c in word):
                    continue
                
                token_text = word.lower() if self.lowercase else word
                
                # Lemmatize
                if self.lemmatizer == 'nltk':
                    wordnet_pos = self._get_wordnet_pos(pos)
                    lemma = self.nltk_lemmatizer.lemmatize(token_text, wordnet_pos)
                else:
                    # Default simple lemmatization
                    lemma = token_text
                
                tokens.append(Token(
                    text=token_text,
                    lemma=lemma,
                    pos=pos,
                    is_asl_lex=False,  # Will be updated later
                    index=i
                ))
            
            if tokens:  # Only add sentences with tokens
                sentences.append(Sentence(tokens=tokens, original_text=sent_text))
        
        return ProcessedText(sentences=sentences)
    
    def process(self, text: str) -> ProcessedText:
        """
        Process input text and return structured representation.
        
        Args:
            text: Input text
            
        Returns:
            ProcessedText object
        """
        if self.tokenizer == 'spacy':
            return self._process_with_spacy(text)
        else:
            return self._process_with_nltk(text)
    
    def process_file(self, file_path: str) -> ProcessedText:
        """
        Process text from a file and return structured representation.
        
        Args:
            file_path: Path to input file
            
        Returns:
            ProcessedText object
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        processed = self.process(text)
        processed.metadata['file_path'] = file_path
        return processed
