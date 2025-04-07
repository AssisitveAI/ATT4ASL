"""
Word similarity measures for the ATT4ASL library.

This module provides different methods for calculating word similarity.
"""

import os
import nltk
from nltk.corpus import wordnet
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
import gensim.downloader as gensim_downloader
from gensim.models import KeyedVectors


class WordEmbeddingModel:
    """
    Word embedding model for semantic similarity.
    """
    
    def __init__(self, model_name: str = 'fasttext', model_path: Optional[str] = None):
        """
        Initialize the word embedding model.
        
        Args:
            model_name: Name of the model ('fasttext', 'word2vec', 'glove')
            model_path: Path to pre-trained model (if None, will download)
        """
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.word_vectors = None
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the word embedding model."""
        if self.model_path and os.path.exists(self.model_path):
            # Load from file
            try:
                self.word_vectors = KeyedVectors.load_word2vec_format(self.model_path)
                return
            except Exception as e:
                print(f"Error loading model from {self.model_path}: {e}")
                print("Falling back to pre-trained models...")
        
        # Map model names to gensim pre-trained models
        model_map = {
            'fasttext': 'fasttext-wiki-news-subwords-300',
            'word2vec': 'word2vec-google-news-300',
            'glove': 'glove-wiki-gigaword-300'
        }
        
        # Use a smaller model if the requested one is not available
        fallback_model = 'glove-twitter-25'
        
        try:
            model_id = model_map.get(self.model_name, fallback_model)
            print(f"Downloading pre-trained {model_id} model...")
            self.word_vectors = gensim_downloader.load(model_id)
        except Exception as e:
            print(f"Error loading pre-trained model {model_id}: {e}")
            print(f"Falling back to {fallback_model}...")
            try:
                self.word_vectors = gensim_downloader.load(fallback_model)
            except Exception as e2:
                print(f"Error loading fallback model: {e2}")
                raise ValueError("Could not load any word embedding model")
    
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Get the vector representation of a word.
        
        Args:
            word: Input word
            
        Returns:
            Word vector or None if not in vocabulary
        """
        word = word.lower()
        try:
            return self.word_vectors[word]
        except KeyError:
            return None
    
    def get_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Similarity score between 0 and 1
        """
        word1, word2 = word1.lower(), word2.lower()
        
        try:
            return self.word_vectors.similarity(word1, word2)
        except KeyError:
            # One or both words not in vocabulary
            return 0.0
    
    def find_most_similar(self, word: str, candidates: List[str], 
                          top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Find the most similar words to a given word from a list of candidates.
        
        Args:
            word: Target word
            candidates: List of candidate words
            top_n: Number of top matches to return
            
        Returns:
            List of (word, similarity) tuples
        """
        word = word.lower()
        word_vector = self.get_word_vector(word)
        
        if word_vector is None:
            return []
        
        similarities = []
        for candidate in candidates:
            candidate = candidate.lower()
            candidate_vector = self.get_word_vector(candidate)
            
            if candidate_vector is not None:
                # Calculate cosine similarity
                similarity = np.dot(word_vector, candidate_vector) / (
                    np.linalg.norm(word_vector) * np.linalg.norm(candidate_vector)
                )
                similarities.append((candidate, float(similarity)))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_n]
    
    def in_vocabulary(self, word: str) -> bool:
        """
        Check if a word is in the model's vocabulary.
        
        Args:
            word: Word to check
            
        Returns:
            True if in vocabulary, False otherwise
        """
        return word.lower() in self.word_vectors


class WordNetInterface:
    """
    Interface to WordNet for synonym finding.
    """
    
    def __init__(self):
        """Initialize the WordNet interface."""
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def get_synonyms(self, word: str, pos: Optional[str] = None) -> List[str]:
        """
        Get synonyms for a word from WordNet.
        
        Args:
            word: Input word
            pos: Part of speech (optional)
            
        Returns:
            List of synonyms
        """
        word = word.lower()
        synonyms = set()
        
        # Convert POS tag to WordNet format
        wordnet_pos = None
        if pos:
            if pos.startswith('N'):
                wordnet_pos = wordnet.NOUN
            elif pos.startswith('V'):
                wordnet_pos = wordnet.VERB
            elif pos.startswith('J'):
                wordnet_pos = wordnet.ADJ
            elif pos.startswith('R'):
                wordnet_pos = wordnet.ADV
        
        # Get synsets
        if wordnet_pos:
            synsets = wordnet.synsets(word, pos=wordnet_pos)
        else:
            synsets = wordnet.synsets(word)
        
        # Extract lemma names (synonyms)
        for synset in synsets:
            for lemma in synset.lemmas():
                synonym = lemma.name().lower().replace('_', ' ')
                if synonym != word:
                    synonyms.add(synonym)
        
        return list(synonyms)
    
    def get_similarity(self, word1: str, word2: str, 
                       pos1: Optional[str] = None, 
                       pos2: Optional[str] = None) -> float:
        """
        Calculate semantic similarity between two words using WordNet.
        
        Args:
            word1: First word
            word2: Second word
            pos1: Part of speech for first word (optional)
            pos2: Part of speech for second word (optional)
            
        Returns:
            Similarity score between 0 and 1
        """
        word1, word2 = word1.lower(), word2.lower()
        
        # Convert POS tags to WordNet format
        wordnet_pos1 = wordnet.NOUN if pos1 and pos1.startswith('N') else None
        wordnet_pos2 = wordnet.NOUN if pos2 and pos2.startswith('N') else None
        
        # Get synsets
        synsets1 = wordnet.synsets(word1, pos=wordnet_pos1) if wordnet_pos1 else wordnet.synsets(word1)
        synsets2 = wordnet.synsets(word2, pos=wordnet_pos2) if wordnet_pos2 else wordnet.synsets(word2)
        
        if not synsets1 or not synsets2:
            return 0.0
        
        # Calculate maximum similarity between any pair of synsets
        max_similarity = 0.0
        for synset1 in synsets1:
            for synset2 in synsets2:
                try:
                    similarity = synset1.path_similarity(synset2)
                    if similarity and similarity > max_similarity:
                        max_similarity = similarity
                except:
                    continue
        
        return max_similarity


class LevenshteinSimilarity:
    """
    Levenshtein distance-based similarity.
    """
    
    @staticmethod
    def get_similarity(word1: str, word2: str) -> float:
        """
        Calculate string similarity based on Levenshtein distance.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Similarity score between 0 and 1
        """
        word1, word2 = word1.lower(), word2.lower()
        
        # Calculate Levenshtein distance
        distance = LevenshteinSimilarity._levenshtein_distance(word1, word2)
        
        # Convert to similarity score
        max_len = max(len(word1), len(word2))
        if max_len == 0:
            return 1.0  # Both strings are empty
        
        return 1.0 - (distance / max_len)
    
    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """
        Calculate Levenshtein distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Levenshtein distance
        """
        if len(s1) < len(s2):
            return LevenshteinSimilarity._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


class SimilarityCalculator:
    """
    Combined similarity calculator using multiple methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the similarity calculator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize similarity measures
        self.use_embeddings = self.config.get('use_embeddings', True)
        self.use_wordnet = self.config.get('use_wordnet', True)
        self.use_levenshtein = self.config.get('use_levenshtein', True)
        
        # Weights for different measures
        self.embedding_weight = self.config.get('embedding_weight', 0.6)
        self.wordnet_weight = self.config.get('wordnet_weight', 0.3)
        self.levenshtein_weight = self.config.get('levenshtein_weight', 0.1)
        
        # Normalize weights
        total_weight = (self.embedding_weight if self.use_embeddings else 0) + \
                       (self.wordnet_weight if self.use_wordnet else 0) + \
                       (self.levenshtein_weight if self.use_levenshtein else 0)
        
        if total_weight > 0:
            if self.use_embeddings:
                self.embedding_weight /= total_weight
            if self.use_wordnet:
                self.wordnet_weight /= total_weight
            if self.use_levenshtein:
                self.levenshtein_weight /= total_weight
        
        # Initialize similarity measures
        if self.use_embeddings:
            model_name = self.config.get('embedding_model', 'fasttext')
            model_path = self.config.get('embedding_path', None)
            self.embedding_model = WordEmbeddingModel(model_name, model_path)
        
        if self.use_wordnet:
            self.wordnet_interface = WordNetInterface()
        
        # POS match weight
        self.pos_match_weight = self.config.get('pos_match_weight', 0.3)
    
    def get_similarity(self, word1: str, word2: str, 
                       pos1: Optional[str] = None, 
                       pos2: Optional[str] = None) -> Tuple[float, Dict[str, float]]:
        """
        Calculate combined similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
            pos1: Part of speech for first word (optional)
            pos2: Part of speech for second word (optional)
            
        Returns:
            Tuple of (combined similarity score, individual scores)
        """
        scores = {}
        
        # Calculate embedding similarity
        if self.use_embeddings:
            embedding_similarity = self.embedding_model.get_similarity(word1, word2)
            scores['embedding'] = embedding_similarity
        else:
            scores['embedding'] = 0.0
        
        # Calculate WordNet similarity
        if self.use_wordnet:
            wordnet_similarity = self.wordnet_interface.get_similarity(word1, word2, pos1, pos2)
            scores['wordnet'] = wordnet_similarity
        else:
            scores['wordnet'] = 0.0
        
        # Calculate Levenshtein similarity
        if self.use_levenshtein:
            levenshtein_similarity = LevenshteinSimilarity.get_similarity(word1, word2)
            scores['levenshtein'] = levenshtein_similarity
        else:
            scores['levenshtein'] = 0.0
        
        # Calculate POS match bonus
        pos_match = 0.0
        if pos1 and pos2 and pos1 == pos2:
            pos_match = 1.0
        scores['pos_match'] = pos_match
        
        # Calculate combined similarity
        combined_similarity = 0.0
        
        if self.use_embeddings:
            combined_similarity += scores['embedding'] * self.embedding_weight
        
        if self.use_wordnet:
            combined_similarity += scores['wordnet'] * self.wordnet_weight
        
        if self.use_levenshtein:
            combined_similarity += scores['levenshtein'] * self.levenshtein_weight
        
        # Apply POS match bonus
        combined_similarity = combined_similarity * (1.0 - self.pos_match_weight) + \
                             pos_match * self.pos_match_weight
        
        return combined_similarity, scores
    
    def find_most_similar(self, word: str, candidates: List[str], 
                          pos: Optional[str] = None, 
                          candidate_pos: Optional[List[str]] = None,
                          top_n: int = 5) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Find the most similar words to a given word from a list of candidates.
        
        Args:
            word: Target word
            candidates: List of candidate words
            pos: Part of speech for target word (optional)
            candidate_pos: List of parts of speech for candidates (optional)
            top_n: Number of top matches to return
            
        Returns:
            List of (word, similarity, scores) tuples
        """
        similarities = []
        
        for i, candidate in enumerate(candidates):
            candidate_p = candidate_pos[i] if candidate_pos and i < len(candidate_pos) else None
            similarity, scores = self.get_similarity(word, candidate, pos, candidate_p)
            similarities.append((candidate, similarity, scores))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_n]
