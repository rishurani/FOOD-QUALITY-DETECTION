"""
LexBrief AI - Configuration Module
AI-Based Court Case Summarization and Legal Research Assistant
Project ID: CSEAIML2026-P13
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    """Application configuration parameters."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'lexbrief-legal-ai-secret-key-2026')
    PROJECT_ID = 'CSEAIML2026-P13'
    PROJECT_NAME = 'LexBrief AI'
    PROJECT_SUBTITLE = 'AI-Based Court Case Summarization and Legal Research Assistant'
    
    DATA_DIR = BASE_DIR / 'data'
    CASES_DIR = DATA_DIR / 'cases'
    CASES_JSON = DATA_DIR / 'cases.json'
    BENCHMARK_JSON = DATA_DIR / 'benchmark_queries.json'
    EVALUATION_JSON = DATA_DIR / 'evaluation_results.json'
    UPLOADS_DIR = BASE_DIR / 'uploads'
    
    # Max file upload: 16 MB
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'json'}
    
    # Search & Summarization Hyperparameters
    DEFAULT_SUMMARY_RATIO = 0.25
    MIN_SUMMARY_SENTENCES = 4
    MAX_SUMMARY_SENTENCES = 15
    BM25_K1 = 1.5
    BM25_B = 0.75
    SEMANTIC_WEIGHT = 0.65
    KEYWORD_WEIGHT = 0.35
