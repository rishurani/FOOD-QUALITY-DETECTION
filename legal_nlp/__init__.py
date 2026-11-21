"""
legal_nlp: Core Natural Language Processing package for Legal Case Summarization
and Precedent Research.
"""
from .preprocessor import LegalPreprocessor
from .summarizer import LegalSummarizer
from .extractor import LegalElementExtractor
from .semantic_search import SemanticSearchEngine
from .comparator import LegalCaseComparator

__all__ = [
    'LegalPreprocessor',
    'LegalSummarizer',
    'LegalElementExtractor',
    'SemanticSearchEngine',
    'LegalCaseComparator',
]
