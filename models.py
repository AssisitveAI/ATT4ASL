"""
Data models for the ATT4ASL library.
"""

from typing import Dict, List, Any, Optional, Tuple


class Token:
    """
    Represents a single token with its attributes.
    """
    
    def __init__(self, text: str, lemma: str, pos: str, is_asl_lex: bool, index: int):
        self.text = text
        self.lemma = lemma
        self.pos = pos
        self.is_asl_lex = is_asl_lex
        self.index = index
    
    def __repr__(self):
        return f"Token(text='{self.text}', lemma='{self.lemma}', pos='{self.pos}', is_asl_lex={self.is_asl_lex})"


class Sentence:
    """
    Represents a sentence with its tokens.
    """
    
    def __init__(self, tokens: List[Token], original_text: str):
        self.tokens = tokens
        self.original_text = original_text
    
    def __repr__(self):
        return f"Sentence(tokens=[{len(self.tokens)} tokens], original_text='{self.original_text[:30]}...')"


class ProcessedText:
    """
    Represents text after preprocessing.
    """
    
    def __init__(self, sentences: List[Sentence], metadata: Optional[Dict[str, Any]] = None):
        self.sentences = sentences
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"ProcessedText(sentences=[{len(self.sentences)} sentences], metadata={self.metadata})"


class AdaptedToken:
    """
    Represents a token after adaptation.
    """
    
    def __init__(self, original_text: str, adapted_text: str, 
                 similarity_score: Optional[float] = None, 
                 replacement_source: Optional[str] = None):
        self.original_text = original_text
        self.adapted_text = adapted_text
        self.was_replaced = original_text != adapted_text
        self.similarity_score = similarity_score
        self.replacement_source = replacement_source
    
    def __repr__(self):
        if self.was_replaced:
            return f"AdaptedToken('{self.original_text}' -> '{self.adapted_text}', score={self.similarity_score:.2f})"
        return f"AdaptedToken('{self.original_text}', unchanged)"


class AdaptedSentence:
    """
    Represents a sentence after adaptation.
    """
    
    def __init__(self, original_tokens: List[Token], adapted_tokens: List[AdaptedToken], 
                 original_text: str, adapted_text: str):
        self.original_tokens = original_tokens
        self.adapted_tokens = adapted_tokens
        self.original_text = original_text
        self.adapted_text = adapted_text
    
    def __repr__(self):
        return f"AdaptedSentence(original='{self.original_text[:30]}...', adapted='{self.adapted_text[:30]}...')"


class AdaptedText:
    """
    Represents text after adaptation.
    """
    
    def __init__(self, original_text: str, adapted_sentences: List[AdaptedSentence], 
                 metadata: Optional[Dict[str, Any]] = None, 
                 stats: Optional[Dict[str, float]] = None):
        self.original_text = original_text
        self.adapted_sentences = adapted_sentences
        self.metadata = metadata or {}
        self.stats = stats or {}
        
        # Compute adapted text
        self.adapted_text = "\n".join([s.adapted_text for s in adapted_sentences])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "original_text": self.original_text,
            "adapted_text": self.adapted_text,
            "stats": self.stats,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        import json
        return json.dumps(self.to_dict(), indent=2)
    
    def to_html(self) -> str:
        """Convert to HTML with highlighting of replacements"""
        html = ["<html><body><h1>ATT4ASL Adaptation</h1>"]
        
        # Original text
        html.append("<h2>Original Text</h2>")
        html.append(f"<div class='original'>{self.original_text}</div>")
        
        # Adapted text
        html.append("<h2>Adapted Text</h2>")
        html.append(f"<div class='adapted'>{self.adapted_text}</div>")
        
        # Detailed comparison
        html.append("<h2>Detailed Comparison</h2>")
        html.append("<table border='1'>")
        html.append("<tr><th>Original</th><th>Adapted</th><th>Similarity</th><th>Source</th></tr>")
        
        for sentence in self.adapted_sentences:
            for token in sentence.adapted_tokens:
                if token.was_replaced:
                    html.append("<tr>")
                    html.append(f"<td>{token.original_text}</td>")
                    html.append(f"<td>{token.adapted_text}</td>")
                    html.append(f"<td>{token.similarity_score:.2f if token.similarity_score else 'N/A'}</td>")
                    html.append(f"<td>{token.replacement_source or 'N/A'}</td>")
                    html.append("</tr>")
        
        html.append("</table>")
        
        # Stats
        if self.stats:
            html.append("<h2>Statistics</h2>")
            html.append("<ul>")
            for key, value in self.stats.items():
                html.append(f"<li><strong>{key}:</strong> {value}</li>")
            html.append("</ul>")
        
        html.append("</body></html>")
        return "\n".join(html)
    
    def __repr__(self):
        return f"AdaptedText(original_length={len(self.original_text)}, adapted_length={len(self.adapted_text)}, stats={self.stats})"
