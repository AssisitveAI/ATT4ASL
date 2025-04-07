"""
ATT4ASL: Speech To Adaptive Text for American Sign Language

A library for converting standard English text into a more accessible form for ASL users
by replacing words not in the ASL-LEX dictionary with similar words that are in the dictionary.
"""

__version__ = "0.1.0"

from .att4asl import ATT4ASL
from .models import AdaptedText, AdaptedSentence, AdaptedToken

__all__ = ["ATT4ASL", "AdaptedText", "AdaptedSentence", "AdaptedToken"]
