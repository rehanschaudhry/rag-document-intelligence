"""
Embedding Engine for RAG System

This module handles:
1. Converting text to embeddings (vectors) using sentence-transformers
2. Building a FAISS vector index for fast similarity search
3. Performing semantic search to find relevant chunks

Key Concepts:
- Embeddings: Converting text into numbers (vectors) that capture meaning
- Semantic Search: Finding similar text based on meaning, not just keywords
- FAISS: Facebook's library for efficient similarity search

Author: Rehan Chaudhry
Project: RAG Document Intelligence System
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from typing import List, Dict, Tuple
import config

class EmbeddingEngine:
    """
    Handles all embedding and similarity search operations
    
    What this does:
    1. Loads all chunks from Parquet files
    2. Converts each chunk's text into a 384-dimensional vector
    3. Builds a FAISS index for fast searching
    4. Finds most similar chunks to a query
    """
    
    def __init__(self):
        """Initialize the embedding engine"""
        
        print("🚀 Initializing Embedding Engine...")
        
        # Load the embedding model (FREE - runs locally!)
        print(f"📥 Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        print(f"✅ Model loaded! Creates {config.EMBEDDING_DIMENSION}-dimensional vectors")
        
        # Storage for chunks and their metadata
        self.chunks = []  # List of chunk dictionaries
        self.embeddings = None  # numpy array of embeddings
        self.index = None  # FAISS index for fast search
        
        # Cache file paths
        self.cache_dir = config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def load_chunks_from_parquet(self, parquet_files: List[str] = None):
        """
        Load all chunks from Parquet files
        
        Args:
            parquet_files: List of parquet file paths (uses config if None)
            
        What happens:
        1. Read each Parquet file
        2. Extract text chunks and metadata
        3. Store in self.chunks
        """
        
        if parquet_files is None:
            parquet_files = config.PARQUET_FILES
        
        print("\n📚 Loading chunks from Parquet files...")
        
        all_chunks = []
        
        for parquet_file in parquet_files:
            if not os.path.exists(parquet_file):
                print(f"⚠️  Warning: {parquet_file} not found, skipping...")
                continue
            
            print(f"   Loading: {parquet_file}")
            
            # Read Parquet file
            df = pd.read_parquet(parquet_file)
            
            # Get document metadata
            doc_info = config.get_document_info(parquet_file)
            
            # Convert each row to a chunk dictionary
            for idx, row in df.iterrows():
                chunk = {
                    'chunk_id': row.get(config.CHUNK_ID_KEY, f"{parquet_file}_{idx}"),
                    'text': row[config.CHUNK_TEXT_KEY],
                    'source_file': parquet_file,
                    'doc_title': doc_info['title'],
                    'doc_authors': doc_info['authors'],
                    'doc_year': doc_info['year'],
                    'doc_topic': doc_info['topic']
                }
                all_chunks.append(chunk)
        
        self.chunks = all_chunks
        print(f"✅ Loaded {len(self.chunks)} chunks from {len(parquet_files)} documents")
        
        return len(self.chunks)
    
    def create_embeddings(self, force_recreate: bool = False):
        """
        Create embeddings for all chunks
        
        Args:
            force_recreate: If True, recreate even if cache exists
            
        What happens:
        1. Check if embeddings are cached
        2. If not cached (or force_recreate), create new embeddings
        3. Convert each chunk's text to a 384-dim vector
        4. Cache the embeddings for faster loading next time
        
        Why embeddings?
        - Converts text to numbers (vectors)
        - Captures semantic meaning
        - Similar text = similar vectors
        - Enables semantic search (meaning-based, not keyword-based)
        """
        
        cache_file = os.path.join(self.cache_dir, "embeddings.npy")
        
        # Check cache first
        if config.CACHE_EMBEDDINGS and os.path.exists(cache_file) and not force_recreate:
            print("\n📦 Loading embeddings from cache...")
            self.embeddings = np.load(cache_file)
            print(f"✅ Loaded {self.embeddings.shape[0]} embeddings from cache")
            return
        
        print("\n🔢 Creating embeddings for all chunks...")
        print(f"   This converts {len(self.chunks)} text chunks into vectors")
        print(f"   Each vector is {config.EMBEDDING_DIMENSION} dimensions")
        
        # Extract just the text from each chunk
        texts = [chunk['text'] for chunk in self.chunks]
        
        # Create embeddings (this is where the magic happens!)
        # The model converts text to vectors that capture meaning
        print("   Processing... (this may take 30-60 seconds)")
        self.embeddings = self.model.encode(
            texts,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"✅ Created {self.embeddings.shape[0]} embeddings!")
        print(f"   Shape: {self.embeddings.shape}")
        
        # Cache for next time
        if config.CACHE_EMBEDDINGS:
            np.save(cache_file, self.embeddings)
            print(f"💾 Cached embeddings to {cache_file}")
    
    def build_index(self):
        """
        Build FAISS index for fast similarity search
        
        What is FAISS?
        - Facebook AI Similarity Search
        - Extremely fast way to find similar vectors
        - Can search millions of vectors in milliseconds
        
        What is an index?
        - Data structure optimized for fast search
        - Like a database index, but for vectors
        - Allows us to quickly find "nearest neighbors"
        
        How it works:
        1. Take all our embeddings (vectors)
        2. Organize them in a special data structure
        3. When we query, quickly find most similar vectors
        """
        
        if self.embeddings is None:
            raise ValueError("Must create embeddings before building index!")
        
        print("\n🔨 Building FAISS index for fast search...")
        
        # Get embedding dimension
        dimension = self.embeddings.shape[1]
        
        # Create FAISS index
        # FlatL2 = exact search using L2 (Euclidean) distance
        # This is simple and accurate, perfect for learning!
        self.index = faiss.IndexFlatL2(dimension)
        
        # Add all embeddings to the index
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"✅ FAISS index built with {self.index.ntotal} vectors")
        print(f"   Ready for semantic search!")
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Perform semantic search to find most relevant chunks
        
        Args:
            query: The question or search query
            top_k: How many results to return (uses config if None)
            
        Returns:
            List of chunk dictionaries with similarity scores
            
        How semantic search works:
        1. Convert query to embedding (vector)
        2. Find chunks with most similar embeddings
        3. Return top_k most similar chunks
        
        Why this is powerful:
        - Finds chunks by MEANING, not just keywords
        - "What is attention?" matches "self-attention mechanism"
        - Even if exact words don't match!
        """
        
        if self.index is None:
            raise ValueError("Must build index before searching!")
        
        if top_k is None:
            top_k = config.TOP_K_CHUNKS
        
        # Convert query to embedding
        # This creates a vector representation of the question
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search FAISS index
        # Returns:
        # - distances: How far each result is (lower = more similar)
        # - indices: Which chunks are most similar
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            top_k
        )
        
        # Convert distances to similarity scores
        # Lower distance = higher similarity
        # We normalize to 0-1 range for easier interpretation
        max_distance = np.max(distances[0]) if len(distances[0]) > 0 else 1.0
        similarities = 1 - (distances[0] / (max_distance + 1e-6))
        
        # Build results
        results = []
        for idx, (chunk_idx, similarity) in enumerate(zip(indices[0], similarities)):
            chunk = self.chunks[chunk_idx].copy()
            chunk['similarity_score'] = float(similarity)
            chunk['rank'] = idx + 1
            chunk['distance'] = float(distances[0][idx])
            results.append(chunk)
        
        # Filter by minimum similarity threshold (but always return at least 1)
        filtered_results = [r for r in results if r['similarity_score'] >= config.MIN_SIMILARITY_SCORE]
        
        # If threshold is too strict, return top result anyway
        if len(filtered_results) == 0 and len(results) > 0:
            filtered_results = [results[0]]
        
        return filtered_results
    
    def initialize_full_pipeline(self):
        """
        Initialize the complete embedding pipeline
        
        This is a convenience method that:
        1. Loads chunks from Parquet
        2. Creates embeddings
        3. Builds FAISS index
        
        Call this once at startup!
        """
        
        print("=" * 60)
        print("INITIALIZING EMBEDDING ENGINE - FULL PIPELINE")
        print("=" * 60)
        
        # Step 1: Load chunks
        num_chunks = self.load_chunks_from_parquet()
        
        if num_chunks == 0:
            raise ValueError("No chunks loaded! Check your Parquet files.")
        
        # Step 2: Create embeddings
        self.create_embeddings()
        
        # Step 3: Build search index
        self.build_index()
        
        print("\n" + "=" * 60)
        print("✅ EMBEDDING ENGINE READY!")
        print("=" * 60)
        print(f"Total chunks indexed: {len(self.chunks)}")
        print(f"Ready for semantic search!")
        print("=" * 60 + "\n")
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the embedding engine
        
        Returns:
            Dictionary with stats
        """
        
        return {
            'total_chunks': len(self.chunks),
            'embedding_dimension': config.EMBEDDING_DIMENSION,
            'model_name': config.EMBEDDING_MODEL_NAME,
            'documents_loaded': len(set(c['source_file'] for c in self.chunks)),
            'index_size': self.index.ntotal if self.index else 0
        }


# ============================================================================
# USAGE EXAMPLE & TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Test the embedding engine
    Run: python embedding_engine.py
    """
    
    print("\n" + "=" * 60)
    print("TESTING EMBEDDING ENGINE")
    print("=" * 60)
    
    # Initialize engine
    engine = EmbeddingEngine()
    
    # Run full pipeline
    engine.initialize_full_pipeline()
    
    # Print stats
    stats = engine.get_stats()
    print("\n📊 STATS:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test search
    print("\n" + "=" * 60)
    print("TESTING SEMANTIC SEARCH")
    print("=" * 60)
    
    test_queries = [
        "What is self-attention?",
        "How does the Transformer architecture work?",
        "What is retrieval-augmented generation?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 60)
        
        results = engine.search(query, top_k=3)
        
        for result in results:
            print(f"\n📄 Rank {result['rank']} - Similarity: {result['similarity_score']:.3f}")
            print(f"   Source: {result['doc_title']}")
            print(f"   Text preview: {result['text'][:150]}...")
    
    print("\n" + "=" * 60)
    print("✅ EMBEDDING ENGINE TEST COMPLETE!")
    print("=" * 60)