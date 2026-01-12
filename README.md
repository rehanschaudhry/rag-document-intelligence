# 🤖 RAG Document Intelligence System

A production-ready Retrieval-Augmented Generation (RAG) system for querying AI research papers using semantic search and Claude AI.

**Live Demo:** Query 122 chunks across 3 foundational AI papers with natural language questions!

## 🎯 What This Does

This RAG system combines semantic search with large language models to answer questions about AI research papers accurately and with citations - preventing hallucinations by grounding responses in actual document content.

**Example Query:**
```
Q: "What is the main innovation of the Transformer architecture?"

A: "The main innovation is computing representations without using 
    sequence-aligned RNNs or convolution [Chunk 1]. Instead, it uses 
    stacked self-attention and point-wise fully connected layers..."
```

## ✨ Key Features

- **Semantic Search**: Finds relevant content by meaning, not just keywords
- **Citation Tracking**: Every claim is cited to source documents
- **Hallucination Prevention**: LLM answers only from provided context
- **Fast Vector Search**: FAISS enables millisecond similarity search
- **Beautiful UI**: Streamlit web interface for easy interaction
- **Production Monitoring**: MLflow integration ready (optional)

## 📚 Documents Indexed

1. **Attention Is All You Need** (Transformer Architecture)
   - Vaswani et al., 2017
   - 45 chunks

2. **Retrieval-Augmented Generation**
   - Lewis et al., 2020
   - 38 chunks

3. **Language Models are Few-Shot Learners** (GPT-3)
   - Brown et al., 2020
   - 39 chunks

**Total:** 122 chunks ready for semantic search

## 🏗️ Architecture

```
User Question
      ↓
Embedding Engine (sentence-transformers)
      ↓
FAISS Vector Search (find similar chunks)
      ↓
RAG Engine (build context prompt)
      ↓
Claude API (generate answer with citations)
      ↓
Answer + Sources + Confidence Metrics
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Anthropic API key ([Get one here](https://console.anthropic.com/))

### Installation

```bash
# Clone repository
git clone https://github.com/rehanschaudhry/rag-document-intelligence.git
cd rag-document-intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Open `config.py`
2. Add your Anthropic API key:
   ```python
   ANTHROPIC_API_KEY = "sk-ant-your-key-here"
   ```
3. Save the file

### Run Web Interface

```bash
streamlit run streamlit_app.py
```

Open browser to `http://localhost:8501` and start asking questions!

### Run Command Line

```bash
python rag_engine.py
```

## 📖 Usage Examples

### Web Interface

1. Launch Streamlit app
2. Enter question in text box
3. Adjust settings in sidebar (optional)
4. Click "Search & Answer"
5. View answer with sources and metrics

### Python API

```python
from rag_engine import RAGEngine

# Initialize
rag = RAGEngine()

# Query
result = rag.query("What is self-attention?")

# Access results
print(result['answer'])        # Claude's answer
print(result['chunks'])        # Source chunks used
print(result['metadata'])      # Confidence metrics
```

## 🔧 Technical Details

### Embedding Model
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Dimensions**: 384
- **Speed**: ~1000 chunks/second
- **Cost**: FREE (runs locally)

### Vector Search
- **Engine**: FAISS (Facebook AI Similarity Search)
- **Index Type**: FlatL2 (exact search)
- **Search Time**: <10ms for 122 chunks

### LLM Generation
- **Model**: Claude Sonnet 4
- **Temperature**: 0.1 (low for factual responses)
- **Max Tokens**: 2000
- **Cost**: ~$0.015 per query

### Hallucination Prevention
- **Retrieval-First**: Always searches documents before generating
- **Citation Requirement**: Forces model to cite sources
- **Context-Only**: Prompts instruct "answer ONLY from context"
- **Faithfulness Tracking**: Monitors semantic similarity to sources

## 📊 Performance Metrics

- **Chunks Indexed**: 122
- **Average Query Time**: 3-5 seconds
- **Search Latency**: <10ms
- **Embedding Cache**: Loads in <1 second after first run
- **Accuracy**: Answers grounded in source documents with citations

## 🗂️ Project Structure

```
rag-document-intelligence/
├── config.py                  # Configuration & API keys
├── embedding_engine.py        # Text → vectors, semantic search
├── rag_engine.py             # Core RAG logic
├── streamlit_app.py          # Web interface
├── requirements.txt          # Dependencies
├── README.md                 # This file
│
├── *.parquet                 # Processed document chunks
├── cache/                    # Cached embeddings
└── venv/                     # Virtual environment
```

## 🎓 How It Works

### 1. Document Processing
PDFs → Text extraction → Chunking → Parquet files

### 2. Embedding Creation
Text chunks → sentence-transformers → 384-dim vectors → Cached to disk

### 3. Index Building
Vectors → FAISS index → Fast similarity search ready

### 4. Query Processing
Question → Embed → Search FAISS → Retrieve top-k chunks

### 5. Answer Generation
Chunks + Question → Build prompt → Claude API → Answer with citations

### 6. Hallucination Prevention
Compare answer to source chunks → Calculate faithfulness → Flag if low confidence

## 🔬 Example Queries

```python
# Transformer Architecture
"What is the main innovation of the Transformer?"
"How does multi-head attention work?"

# RAG Systems
"What problem does RAG solve?"
"How does RAG prevent hallucinations?"

# GPT-3
"How does GPT-3 perform few-shot learning?"
"What are the limitations of large language models?"
```

## 🛠️ Customization

### Add Your Own Documents

1. Process PDFs into Parquet chunks
2. Add to `config.PARQUET_FILES`
3. Add metadata to `config.DOCUMENT_METADATA`
4. Run `python embedding_engine.py` to rebuild index

### Adjust Retrieval Settings

```python
# In config.py
TOP_K_CHUNKS = 5              # Retrieve more chunks
MIN_SIMILARITY_SCORE = 0.2    # Lower threshold
```

### Modify Prompt Template

Edit `build_rag_prompt()` in `rag_engine.py` to customize how context is provided to Claude.

## 📈 Future Enhancements

- [ ] Add MLflow hallucination tracking
- [ ] Support for more document types (Word, HTML)
- [ ] Multi-modal support (images, tables)
- [ ] Query history and conversation context
- [ ] Advanced chunking strategies
- [ ] Hybrid search (semantic + keyword)

## 🤝 Contributing

This is a portfolio project, but suggestions and feedback are welcome!

## 📄 License

MIT License - feel free to use for learning and portfolio projects

## 👤 Author

**Rehan Chaudhry**
- Data Scientist | ML Engineer
- Building production ML systems
- [GitHub](https://github.com/rehanschaudhry)

## 🙏 Acknowledgments

- Papers: Vaswani et al. (Transformer), Lewis et al. (RAG), Brown et al. (GPT-3)
- Models: sentence-transformers, Claude by Anthropic
- Tools: FAISS (Meta), Streamlit, MLflow

---

**Built with:** Python • Claude API • sentence-transformers • FAISS • Streamlit

**Skills Demonstrated:** RAG Systems • Vector Databases • LLM Integration • Semantic Search • Production ML • Hallucination Prevention
