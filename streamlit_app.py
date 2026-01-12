"""
Streamlit Web Interface for RAG Document Intelligence System

This provides a simple, beautiful web UI for your RAG system.

Run with: streamlit run streamlit_app.py

Author: Rehan Chaudhry
Project: RAG Document Intelligence System
"""

import streamlit as st
from rag_engine import RAGEngine
import config

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT
)

# Initialize RAG engine (cached so it only runs once)
@st.cache_resource
def load_rag_engine():
    """Load and cache the RAG engine"""
    return RAGEngine()

# Main app
def main():
    """Main Streamlit app"""
    
    # Title
    st.title("🤖 RAG Document Intelligence System")
    st.markdown("*Ask questions about AI research papers - powered by Claude & semantic search*")
    
    st.markdown("---")
    
    # Initialize RAG engine
    with st.spinner("🚀 Initializing RAG system..."):
        rag = load_rag_engine()
    
    # Sidebar with info
    with st.sidebar:
        st.header("📚 Available Documents")
        st.markdown("""
        - **Attention Is All You Need** (Transformer)
          - Vaswani et al., 2017
        
        - **Retrieval-Augmented Generation**
          - Lewis et al., 2020
        
        - **Language Models are Few-Shot Learners** (GPT-3)
          - Brown et al., 2020
        """)
        
        st.markdown("---")
        
        st.header("⚙️ Settings")
        
        # Number of chunks to retrieve
        top_k = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=5,
            value=config.TOP_K_CHUNKS,
            help="How many relevant chunks to find"
        )
        
        # Temperature
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=config.CLAUDE_TEMPERATURE,
            step=0.1,
            help="Lower = more focused, Higher = more creative"
        )
        
        st.markdown("---")
        
        st.header("ℹ️ About")
        st.markdown("""
        This RAG system:
        - Searches 122 chunks from 3 AI papers
        - Uses semantic search (not keywords!)
        - Prevents hallucinations with citations
        - Powered by Claude Sonnet 4
        """)
    
    # Main interface
    st.header("💬 Ask a Question")
    
    # Example questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - What is the main innovation of the Transformer architecture?
        - How does self-attention work?
        - What problem does RAG solve?
        - How does GPT-3 perform few-shot learning?
        - What are the limitations of large language models?
        - How does multi-head attention work?
        """)
    
    # Question input
    question = st.text_input(
        "Your question:",
        placeholder="e.g., What is self-attention?",
        key="question_input"
    )
    
    # Search button
    if st.button("🔍 Search & Answer", type="primary"):
        if not question:
            st.warning("⚠️ Please enter a question!")
        else:
            # Process query
            with st.spinner("🔍 Searching documents and generating answer..."):
                result = rag.query(
                    question=question,
                    top_k=top_k,
                    temperature=temperature
                )
            
            # Display answer
            st.markdown("---")
            st.subheader("💡 Answer")
            st.markdown(result['answer'])
            
            # Display sources
            if config.SHOW_RETRIEVED_CHUNKS:
                st.markdown("---")
                st.subheader("📚 Sources Used")
                
                for i, chunk in enumerate(result['chunks'], 1):
                    with st.expander(f"📄 Source {i}: {chunk['doc_title']} (Similarity: {chunk['similarity_score']:.3f})"):
                        st.markdown(f"**Authors:** {chunk['doc_authors']} ({chunk['doc_year']})")
                        st.markdown(f"**Topic:** {chunk['doc_topic']}")
                        st.markdown(f"**Similarity Score:** {chunk['similarity_score']:.3f}")
                        st.markdown("---")
                        st.markdown("**Text:**")
                        st.text(chunk['text'])
            
            # Display metrics
            if config.SHOW_METRICS:
                st.markdown("---")
                st.subheader("📊 Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Chunks Retrieved",
                        result['metadata']['chunks_found']
                    )
                
                with col2:
                    st.metric(
                        "Avg Similarity",
                        f"{result['metadata']['avg_similarity']:.3f}"
                    )
                
                with col3:
                    st.metric(
                        "Top Similarity",
                        f"{result['metadata']['top_similarity']:.3f}"
                    )

if __name__ == "__main__":
    main()
