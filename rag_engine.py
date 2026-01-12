"""
RAG Engine - Retrieval-Augmented Generation

This is the CORE of your RAG system!

What it does:
1. Takes a user's question
2. Uses EmbeddingEngine to find relevant chunks
3. Builds a prompt with those chunks as context
4. Sends to Claude API for answer generation
5. Returns answer with citations

This is where retrieval (finding info) meets generation (creating answers)!

Author: Rehan Chaudhry
Project: RAG Document Intelligence System
"""

from anthropic import Anthropic
from typing import List, Dict, Optional
import config
from embedding_engine import EmbeddingEngine

class RAGEngine:
    """
    Retrieval-Augmented Generation Engine
    
    This combines:
    - Semantic search (finding relevant chunks)
    - LLM generation (creating intelligent answers)
    - Citation tracking (ensuring faithfulness)
    """
    
    def __init__(self):
        """Initialize the RAG engine"""
        
        print("🚀 Initializing RAG Engine...")
        
        # Initialize Claude client
        self.client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        print(f"✅ Claude API client ready (model: {config.CLAUDE_MODEL})")
        
        # Initialize embedding engine
        print("\n📚 Setting up embedding engine...")
        self.embedding_engine = EmbeddingEngine()
        self.embedding_engine.initialize_full_pipeline()
        
        print("\n✅ RAG Engine fully initialized and ready!")
    
    def build_rag_prompt(self, question: str, chunks: List[Dict]) -> str:
        """
        Build a prompt for Claude with retrieved chunks as context
        
        Args:
            question: User's question
            chunks: Retrieved chunks from semantic search
            
        Returns:
            Formatted prompt string
            
        Why this matters:
        - Gives Claude ACTUAL text from your documents
        - Prevents hallucination (Claude answers from provided context)
        - Enables citations (chunks have IDs)
        
        Prompt engineering principles used:
        1. Clear instructions ("Answer ONLY from context")
        2. Structured context (numbered chunks)
        3. Citation requirement (forces grounding)
        4. Fallback behavior (what to do if answer not in context)
        """
        
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            # Format each chunk with metadata
            chunk_text = f"""
[Chunk {i}]
Source: {chunk['doc_title']} ({chunk['doc_authors']}, {chunk['doc_year']})
Similarity: {chunk['similarity_score']:.3f}
Text: {chunk['text']}
"""
            context_parts.append(chunk_text)
        
        full_context = "\n".join(context_parts)
        
        # Build the complete prompt
        prompt = f"""You are an AI assistant helping users understand research papers on AI and machine learning.

You have been given relevant excerpts from academic papers. Your task is to answer the question based ONLY on the provided context.

CONTEXT FROM RESEARCH PAPERS:
{full_context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer the question using ONLY information from the chunks above
2. Cite your sources using [Chunk X] notation after each claim
3. If the answer requires information from multiple chunks, synthesize them coherently
4. If the chunks don't contain enough information to fully answer the question, say so clearly
5. Do not add information from your general knowledge - stick to the provided context
6. Be concise but thorough

ANSWER:"""
        
        return prompt
    
    def query(self, 
              question: str, 
              top_k: Optional[int] = None,
              temperature: Optional[float] = None,
              max_tokens: Optional[int] = None) -> Dict:
        """
        Main RAG query method
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve (uses config default if None)
            temperature: Claude temperature (uses config default if None)
            max_tokens: Max response length (uses config default if None)
            
        Returns:
            Dictionary with:
            - answer: Claude's response
            - chunks: Chunks that were used
            - question: Original question
            - metadata: Additional info
            
        This is the MAIN function - it orchestrates everything!
        """
        
        print(f"\n{'='*60}")
        print(f"🔍 Processing Question: '{question}'")
        print(f"{'='*60}")
        
        # Step 1: Retrieve relevant chunks
        print("\n📚 Step 1: Retrieving relevant chunks...")
        chunks = self.embedding_engine.search(
            query=question,
            top_k=top_k or config.TOP_K_CHUNKS
        )
        
        print(f"✅ Retrieved {len(chunks)} chunks")
        for i, chunk in enumerate(chunks, 1):
            print(f"   {i}. {chunk['doc_title']} (similarity: {chunk['similarity_score']:.3f})")
        
        # Check if we found any chunks
        if not chunks:
            return {
                'question': question,
                'answer': "I couldn't find any relevant information in the documents to answer this question.",
                'chunks': [],
                'metadata': {
                    'chunks_found': 0,
                    'model': config.CLAUDE_MODEL,
                    'error': 'No chunks above similarity threshold'
                }
            }
        
        # Step 2: Build prompt with context
        print("\n📝 Step 2: Building RAG prompt...")
        prompt = self.build_rag_prompt(question, chunks)
        print(f"✅ Prompt built ({len(prompt)} characters)")
        
        # Step 3: Call Claude API
        print("\n🤖 Step 3: Generating answer with Claude...")
        try:
            response = self.client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=max_tokens or config.CLAUDE_MAX_TOKENS,
                temperature=temperature or config.CLAUDE_TEMPERATURE,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            answer = response.content[0].text
            print(f"✅ Answer generated ({len(answer)} characters)")
            
        except Exception as e:
            print(f"❌ Error calling Claude API: {str(e)}")
            return {
                'question': question,
                'answer': f"Error generating answer: {str(e)}",
                'chunks': chunks,
                'metadata': {
                    'chunks_found': len(chunks),
                    'model': config.CLAUDE_MODEL,
                    'error': str(e)
                }
            }
        
        # Step 4: Return results
        result = {
            'question': question,
            'answer': answer,
            'chunks': chunks,
            'metadata': {
                'chunks_found': len(chunks),
                'model': config.CLAUDE_MODEL,
                'temperature': temperature or config.CLAUDE_TEMPERATURE,
                'max_tokens': max_tokens or config.CLAUDE_MAX_TOKENS,
                'avg_similarity': sum(c['similarity_score'] for c in chunks) / len(chunks),
                'top_similarity': chunks[0]['similarity_score'] if chunks else 0
            }
        }
        
        print(f"\n{'='*60}")
        print("✅ RAG QUERY COMPLETE")
        print(f"{'='*60}\n")
        
        return result
    
    def print_result(self, result: Dict):
        """
        Pretty-print a RAG result
        
        Args:
            result: Result dictionary from query()
        """
        
        print("\n" + "="*60)
        print("RAG QUERY RESULT")
        print("="*60)
        
        print(f"\n❓ QUESTION:")
        print(f"   {result['question']}")
        
        print(f"\n💡 ANSWER:")
        print(f"   {result['answer']}")
        
        print(f"\n📚 SOURCES USED ({len(result['chunks'])} chunks):")
        for i, chunk in enumerate(result['chunks'], 1):
            print(f"\n   [{i}] {chunk['doc_title']}")
            print(f"       Authors: {chunk['doc_authors']} ({chunk['doc_year']})")
            print(f"       Similarity: {chunk['similarity_score']:.3f}")
            print(f"       Preview: {chunk['text'][:100]}...")
        
        print(f"\n📊 METADATA:")
        for key, value in result['metadata'].items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
        print("\n" + "="*60 + "\n")


# ============================================================================
# USAGE EXAMPLE & TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Test the RAG engine
    Run: python rag_engine.py
    
    This will:
    1. Initialize the RAG system
    2. Ask test questions
    3. Show you how RAG works!
    """
    
    print("\n" + "="*60)
    print("TESTING RAG ENGINE")
    print("="*60)
    
    # Initialize RAG engine
    rag = RAGEngine()
    
    # Test questions
    test_questions = [
        "What is the main innovation of the Transformer architecture?",
        "How does self-attention work?",
        "What problem does RAG solve?",
    ]
    
    # Process each question
    for question in test_questions:
        result = rag.query(question)
        rag.print_result(result)
        
        # Wait for user to read
        input("\nPress Enter to continue to next question...")
    
    print("\n" + "="*60)
    print("✅ RAG ENGINE TEST COMPLETE!")
    print("="*60)
    print("\nYour RAG system is working!")
    print("Next: Build Streamlit UI and add MLflow tracking!")
    print("="*60 + "\n")
