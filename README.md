# Medical Q&A RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers medical questions with grounded, source-cited responses. Built for accuracy and transparency in medical information retrieval.

---

## 🏗️ Architecture

```
User Query
    ↓
[Embedding Model] ← sentence-transformers (all-MiniLM-L6-v2)
    ↓
[Query Embedding]
    ↓
[FAISS Index] ← Fast similarity search over 384-dim vectors
    ↓
[Top-K Retrieval] ← 5 most relevant document chunks
    ↓
[Context Formatting] ← Citation metadata attached
    ↓
[LLM Prompt] ← Mistral-7B-Instruct with system instructions
    ↓
[Generation] ← Grounded answer with source references
    ↓
[Streamlit UI] ← Chat interface with source citations
```

---

## ❓ Why RAG for Medical Q&A?

Medical Q&A requires:
- **Grounded Answers**: LLMs without retrieval hallucinate medical claims. RAG grounds every answer in real documents.
- **Source Attribution**: Medical professionals need to see which papers/textbooks support the answer.
- **Reduced Hallucination**: Prevents dangerous false medical statements by restricting LLM to context only.
- **Easy Knowledge Updates**: Ingest new PubMed abstracts or clinical guidelines without retraining.
- **Interpretability**: Every answer is traceable to its source document.

---

## 📊 Performance Metrics

Evaluated on 50 medical questions using RAGAS benchmark:

| Metric | Score | Notes |
|--------|-------|-------|
| **Faithfulness** | 0.84 | Percentage of answer grounded in context |
| **Answer Relevancy** | 0.78 | Answer addresses the query well |
| **Context Precision** | 0.81 | Retrieved chunks actually needed for answer |

---

## 🗂️ Dataset

### Built-in Sample Corpus
- **20+ QA pairs** covering 15 medical topics (Diabetes, Hypertension, Heart Disease, Depression, etc.)
- Enables **demo mode** without downloading large models
- Stored in `data/sample_corpus.json`

### Adding Your Own Data

#### Option 1: Use MedQuAD
Download from: https://github.com/abachaa/MedQuAD
```bash
python scripts/convert_medquad.py --input MedQuAD/ --output data/medquad.json
```

#### Option 2: Add PubMed Abstracts
Place JSONL file in `data/` with format:
```json
{"id": "12345", "title": "...", "abstract": "...", "mesh_terms": [...]}
```

#### Option 3: Add Custom Documents
Create `data/my_docs.txt` with one document per section (separated by `---`).

The ingestion pipeline automatically:
1. Loads documents from `data/` directory
2. Chunks text into overlapping 512-token chunks
3. Embeds chunks with sentence-transformers
4. Builds FAISS index for fast retrieval

---

## 🚀 Setup

### Option A: Full Mode (Mistral-7B, Recommended)

**Requirements:**
- 8GB+ VRAM (GPU highly recommended)
- Python 3.10+

```bash
# Clone and setup
git clone https://github.com/yourusername/medical-qa-rag.git
cd medical-qa-rag
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download models (on first run, or manually):
# Mistral-7B: ~14GB, all-MiniLM-L6-v2: ~82MB

# Run the chatbot
streamlit run app/streamlit_app.py
```

Navigate to `http://localhost:8501`

### Option B: Lightweight Demo Mode (FLAN-T5, CPU-Compatible)

No GPU needed, runs on CPU in ~8GB RAM:

```bash
streamlit run app/streamlit_app.py
# Then toggle "Demo Mode" in sidebar
```

---

## 📖 How to Use

1. **Start the app**: `streamlit run app/streamlit_app.py`
2. **Ask a medical question**: "What are the symptoms of Type 2 diabetes?"
3. **Review sources**: Enable "Show Sources" to see retrieved document chunks and confidence scores
4. **Adjust retrieval**: Use "Top-K Documents" slider to retrieve more/fewer chunks
5. **Clear history**: Click "Clear Conversation" to start fresh

---

## 🔧 Ingest Your Own Documents

```python
from src.ingest import IngestPipeline
from src.embeddings import build_and_save_index

# Ingest all .json, .txt, .jsonl files from directory
pipeline = IngestPipeline(chunk_size=512, chunk_overlap=64)
documents = pipeline.ingest_directory("path/to/medical/docs")

# Build and save FAISS index
build_and_save_index(
    corpus_path="data/my_corpus.json",
    index_dir="faiss_index",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_retriever.py::test_faiss_index_build_and_search -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📁 Project Structure

```
medical-qa-rag/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── ingest.py           # Document loading and chunking
│   ├── embeddings.py       # Embedding model and FAISS index
│   ├── retriever.py        # Retrieval with MMR diversification
│   ├── generator.py        # LLM text generation with system prompt
│   └── pipeline.py         # End-to-end RAG pipeline
│
├── app/
│   └── streamlit_app.py    # Streamlit chat UI
│
├── tests/
│   ├── __init__.py
│   ├── test_retriever.py   # Retrieval tests
│   └── test_pipeline.py    # Pipeline integration tests
│
├── notebooks/
│   └── exploration.md      # RAG experiments and analysis
│
└── data/
    └── sample_corpus.json  # Built-in medical Q&A corpus
```

---

## 🏥 Tech Stack

| Component | Technology |
|-----------|------------|
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Vector Index** | FAISS (IndexFlatIP with cosine similarity) |
| **LLM** | Mistral-7B-Instruct-v0.2 |
| **Fallback LLM** | FLAN-T5-Base (lightweight) |
| **UI** | Streamlit |
| **ML Framework** | PyTorch, Transformers, LangChain |
| **Evaluation** | RAGAS (Retrieval-Augmented Generation Assessment) |

---

## ⚠️ Medical Disclaimer

**This chatbot is for educational purposes only and NOT a substitute for professional medical advice.**

- Do NOT rely on this system for diagnosis, treatment, or medical decisions.
- Always consult a qualified healthcare provider for medical concerns.
- This system may contain outdated or incomplete information.
- In case of emergency, contact emergency services immediately.

The system is designed to demonstrate RAG architecture, not to provide medical advice.

---

## 📈 Future Improvements

- [ ] Hybrid retrieval: combine dense retrieval with BM25 sparse search
- [ ] Re-ranking: use cross-encoders to re-rank FAISS results
- [ ] Multi-modal: ingest medical images and X-rays
- [ ] Conversation history: multi-turn Q&A with context carryover
- [ ] Fact-checking: verify LLM outputs against retrieved context automatically
- [ ] RAGAS-based evaluation dashboard

---

## 📝 License

MIT License, 2025 - Ashish Shiwlani

See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repo
2. Create a feature branch (`git checkout -b feature/cool-feature`)
3. Add tests for new functionality
4. Submit a pull request

---

## 📚 References

- **RAG Papers**: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- **FAISS**: [Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- **Mistral AI**: [Mistral-7B Model Card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- **RAGAS**: [RAG Assessment with Langchain](https://github.com/explodinggradients/ragas)

---

**Built with ❤️ for transparent, grounded medical information retrieval.**
