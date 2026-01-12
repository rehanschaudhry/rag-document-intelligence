"""
Configuration file for RAG Document Intelligence System

This file stores all API keys, model settings, and configuration parameters.
IMPORTANT: Never commit this file to GitHub with real API keys!
Add config.py to .gitignore after adding your keys.

Author: Rehan Chaudhry
Project: RAG Document Intelligence System
"""

import os

# ============================================================================
# API KEYS
# ============================================================================

# Anthropic Claude API Key
# Get from: https://console.anthropic.com/
# Used for: LLM answer generation
ANTHROPIC_API_KEY = "your-anthropic-api-key-here"  # Replace with your actual key

# NOTE: We're using FREE sentence-transformers for embeddings
# No OpenAI API key needed! This saves you money while learning.

# ============================================================================
# MODEL SETTINGS
# ============================================================================

# Claude Model Configuration
CLAUDE_MODEL = "claude-sonnet-4-20250514"  # Latest Sonnet model
CLAUDE_MAX_TOKENS = 2000  # Max length of Claude's response
CLAUDE_TEMPERATURE = 0.1  # Low temperature = more focused, less creative
                          # Range: 0.0 (deterministic) to 1.0 (creative)
                          # For factual Q&A, keep it low (0.1-0.3)

# Embedding Model Configuration (FREE - runs locally!)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # sentence-transformers model
                                            # Creates 384-dim vectors
                                            # Fast, accurate, FREE!
EMBEDDING_DIMENSION = 384  # Dimension of embedding vectors

# ============================================================================
# RAG SETTINGS
# ============================================================================

# Retrieval Settings
TOP_K_CHUNKS = 3  # How many chunks to retrieve for each question
                  # More chunks = more context but slower & more tokens
                  # Sweet spot: 3-5 chunks

MIN_SIMILARITY_SCORE = 0.3  # Minimum similarity to consider chunk relevant
                             # Range: 0.0 to 1.0
                             # Lower = more lenient, higher = more strict

# Chunk Processing
CHUNK_TEXT_KEY = "text"  # Column name in Parquet with chunk text
CHUNK_ID_KEY = "chunk_id"  # Column name with chunk ID

# ============================================================================
# HALLUCINATION DETECTION SETTINGS
# ============================================================================

# Faithfulness Thresholds
FAITHFULNESS_THRESHOLD_HIGH = 0.8  # >0.8 = high confidence
FAITHFULNESS_THRESHOLD_LOW = 0.6   # <0.6 = potential hallucination

# Citation Requirements
REQUIRE_CITATIONS = True  # Force model to cite sources
CITATION_FORMAT = "[Chunk {chunk_id}]"  # How citations should look

# ============================================================================
# MLFLOW SETTINGS
# ============================================================================

# MLflow Tracking
MLFLOW_TRACKING_URI = "file:./mlruns"  # Local MLflow storage
MLFLOW_EXPERIMENT_NAME = "RAG_Document_Intelligence"  # Experiment name

# What to Log
LOG_QUESTIONS = True  # Log every question asked
LOG_ANSWERS = True  # Log every answer generated
LOG_CHUNKS = True  # Log which chunks were retrieved
LOG_METRICS = True  # Log hallucination metrics

# ============================================================================
# DATA PATHS
# ============================================================================

# Parquet Files (your processed PDFs)
PARQUET_FILES = [
    "1706.03762v7_chunks.parquet",  # Transformer paper
    "2005.11401v4_chunks.parquet",  # RAG paper
    "2005.14165v4_chunks.parquet",  # GPT-3 paper
]

# Metadata for each document (for better citations)
DOCUMENT_METADATA = {
    "1706.03762v7_chunks.parquet": {
        "title": "Attention Is All You Need (Transformer)",
        "authors": "Vaswani et al.",
        "year": 2017,
        "topic": "Transformer Architecture"
    },
    "2005.11401v4_chunks.parquet": {
        "title": "Retrieval-Augmented Generation (RAG)",
        "authors": "Lewis et al.",
        "year": 2020,
        "topic": "RAG Systems"
    },
    "2005.14165v4_chunks.parquet": {
        "title": "Language Models are Few-Shot Learners (GPT-3)",
        "authors": "Brown et al.",
        "year": 2020,
        "topic": "Large Language Models"
    }
}

# ============================================================================
# STREAMLIT UI SETTINGS
# ============================================================================

# UI Configuration
PAGE_TITLE = "RAG Document Intelligence"
PAGE_ICON = "🤖"
LAYOUT = "wide"

# Display Settings
SHOW_RETRIEVED_CHUNKS = True  # Show chunks used in answer
SHOW_METRICS = True  # Show hallucination metrics
SHOW_MLFLOW_LINK = True  # Show link to MLflow experiment

# ============================================================================
# ADVANCED SETTINGS (Usually don't need to change)
# ============================================================================

# FAISS Index Settings
FAISS_INDEX_TYPE = "FlatL2"  # L2 distance (Euclidean)
                             # Alternative: "FlatIP" for cosine similarity
                             # For learning: FlatL2 is fine

# Batch Processing
EMBEDDING_BATCH_SIZE = 32  # Process this many chunks at once
                           # Larger = faster but more memory

# Cache Settings
CACHE_EMBEDDINGS = True  # Cache embeddings to disk (faster restarts)
CACHE_DIR = "./cache"  # Where to store cached embeddings

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_config():
    """
    Validate that all required configuration is set
    Returns: (bool, str) - (is_valid, error_message)
    """
    
    # Check API key
    if not ANTHROPIC_API_KEY or not ANTHROPIC_API_KEY.startswith("sk-ant-"):
        return False, "Please set your ANTHROPIC_API_KEY in config.py"
    
    # Check Parquet files exist
    import os
    for parquet_file in PARQUET_FILES:
        if not os.path.exists(parquet_file):
            return False, f"Parquet file not found: {parquet_file}"
    
    return True, "Configuration valid"


def get_document_info(parquet_filename):
    """
    Get metadata for a document
    
    Args:
        parquet_filename: Name of the parquet file
        
    Returns:
        dict: Document metadata
    """
    return DOCUMENT_METADATA.get(parquet_filename, {
        "title": "Unknown Document",
        "authors": "Unknown",
        "year": "Unknown",
        "topic": "Unknown"
    })


def print_config():
    """Print current configuration (for debugging)"""
    
    print("=" * 60)
    print("RAG SYSTEM CONFIGURATION")
    print("=" * 60)
    print(f"Claude Model: {CLAUDE_MODEL}")
    print(f"Temperature: {CLAUDE_TEMPERATURE}")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"Embedding Dimension: {EMBEDDING_DIMENSION}")
    print(f"Top-K Chunks: {TOP_K_CHUNKS}")
    print(f"Parquet Files: {len(PARQUET_FILES)}")
    for pf in PARQUET_FILES:
        info = get_document_info(pf)
        print(f"  - {info['title']} ({info['year']})")
    print(f"MLflow Experiment: {MLFLOW_EXPERIMENT_NAME}")
    print("=" * 60)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Test configuration by running: python config.py
    """
    
    # Print configuration
    print_config()
    
    # Validate configuration
    is_valid, message = validate_config()
    
    if is_valid:
        print("\n✅ Configuration is valid!")
        print("Ready to build RAG system!")
    else:
        print(f"\n❌ Configuration error: {message}")
        print("\nPlease fix the issue and try again.")