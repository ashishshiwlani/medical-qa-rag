"""
Streamlit chat UI for Medical Q&A RAG Chatbot.

Features:
  - Multi-turn conversation with persistent history
  - Source citations with similarity scores
  - Configurable retrieval (top-k slider)
  - Mode toggle (Demo vs Full)
  - Medical disclaimer banner
  - Beautiful Streamlit styling

Key concepts:
  - session_state: Persistent conversation history across re-runs
  - @st.cache_resource: Pipeline loaded once and reused (expensive operation)
  - chat_input/chat_message: High-level Streamlit chat components
"""

import streamlit as st
import time
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline


# ============================================================================
# Streamlit Configuration
# ============================================================================

st.set_page_config(
    page_title="Medical Q&A RAG Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add CSS for custom styling
st.markdown("""
<style>
    .medical-disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 4px;
        padding: 12px;
        margin-bottom: 20px;
        font-size: 0.9em;
    }
    .source-citation {
        background-color: #f8f9fa;
        border-left: 3px solid #0d6efd;
        padding: 8px;
        margin: 8px 0;
        border-radius: 3px;
    }
    .score-badge {
        display: inline-block;
        background-color: #e7f3ff;
        color: #0066cc;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Initialize Session State
# ============================================================================

def initialize_session_state():
    """Set up session state variables on first page load."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pipeline_initialized" not in st.session_state:
        st.session_state.pipeline_initialized = False


initialize_session_state()


# ============================================================================
# Pipeline Caching
# ============================================================================

@st.cache_resource
def load_pipeline(use_demo_mode: bool = True):
    """
    Load RAG pipeline once and cache it.

    st.cache_resource ensures:
      - Pipeline loaded only once per session
      - Same instance reused across re-runs
      - Major speedup for subsequent queries

    Args:
        use_demo_mode: Whether to use lightweight models

    Returns:
        Initialized RAGPipeline instance
    """
    with st.spinner("Loading RAG pipeline... (this may take 1-2 minutes)"):
        pipeline = RAGPipeline(
            index_dir="faiss_index",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            llm_model_name="mistralai/Mistral-7B-Instruct-v0.2",
            top_k=5,
            use_demo_mode=use_demo_mode
        )
    return pipeline


# ============================================================================
# Main UI Layout
# ============================================================================

# Medical Disclaimer (sticky at top)
st.markdown("""
<div class="medical-disclaimer">
    ⚠️ <strong>Medical Disclaimer:</strong> This chatbot is for educational purposes only.
    It is NOT a substitute for professional medical advice, diagnosis, or treatment.
    Always consult a qualified healthcare provider for medical concerns.
    In case of emergency, contact emergency services immediately.
</div>
""", unsafe_allow_html=True)

# Page title
st.title("🏥 Medical Q&A RAG Chatbot")
st.markdown(
    "Ask medical questions and get grounded answers with source citations. "
    "Powered by Retrieval-Augmented Generation (RAG)."
)


# ============================================================================
# Sidebar Controls
# ============================================================================

with st.sidebar:
    st.header("⚙️ Settings")

    # Mode selection
    use_demo_mode = st.toggle(
        "Demo Mode (Lightweight)",
        value=True,
        help="Use CPU-friendly FLAN-T5 instead of Mistral-7B. Faster but lower quality."
    )

    # Reload pipeline if mode changed
    if st.button("🔄 Reload Pipeline", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    # Retrieval settings
    st.subheader("Retrieval")
    top_k = st.slider(
        "Number of Documents to Retrieve",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="More documents = more context but slower retrieval"
    )

    show_sources = st.checkbox(
        "Show Source Citations",
        value=True,
        help="Display retrieved documents and similarity scores"
    )

    st.divider()

    # Conversation control
    st.subheader("Conversation")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.success("Conversation cleared!")
        st.rerun()

    st.divider()

    # About section
    st.subheader("ℹ️ About")
    st.markdown("""
    **Medical Q&A RAG Chatbot**

    - **Retrieval**: FAISS with sentence-transformers
    - **Generation**: Mistral-7B (full) or FLAN-T5 (demo)
    - **Architecture**: Retrieval-Augmented Generation (RAG)

    [GitHub](https://github.com) | [Docs](https://github.com)
    """)

    # Model info
    st.caption(f"Mode: {'Demo (CPU)' if use_demo_mode else 'Full (GPU)'}")


# ============================================================================
# Main Chat Interface
# ============================================================================

# Load pipeline with selected mode
pipeline = load_pipeline(use_demo_mode=use_demo_mode)

# Display conversation history
chat_container = st.container(height=500, border=True)

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources for assistant messages if enabled
            if message["role"] == "assistant" and show_sources:
                if "sources" in message:
                    with st.expander("📚 View Sources", expanded=False):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"""
<div class="source-citation">
    <strong>[{i}]</strong>
    <span class="score-badge">Similarity: {source.score:.2f}</span><br/>
    <em>Source: {source.document.metadata.get('source', 'Unknown')}</em>
    {f"| Topic: {source.document.metadata.get('topic', '')}" if source.document.metadata.get('topic') else ""}
    <br/><br/>
    {source.document.content[:300]}...
</div>
""", unsafe_allow_html=True)


# ============================================================================
# Chat Input and Processing
# ============================================================================

# Chat input at bottom
user_input = st.chat_input("Ask a medical question...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Display user message immediately
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)

    # Generate response with token-by-token streaming
    with chat_container:
        with st.chat_message("assistant"):
            try:
                # ── Phase 1: Retrieval (blocking, ~100ms) ──────────────────────
                with st.spinner("🔍 Retrieving relevant medical information..."):
                    sources, context = pipeline.retrieve(user_input)

                # ── Phase 2: Streaming generation ─────────────────────────────
                # st.write_stream() consumes the iterator and renders tokens
                # as they arrive, giving a ChatGPT-like typing effect.
                # We wrap the generator so we can capture the full text for
                # storing in session_state afterwards.

                full_answer_parts: list = []

                def _token_iter():
                    for token in pipeline._stream_from_context(user_input, context):
                        full_answer_parts.append(token)
                        yield token

                if context:
                    full_answer = st.write_stream(_token_iter())
                else:
                    fallback = "I don't have information about that in the available medical literature."
                    st.markdown(fallback)
                    full_answer = fallback

                # ── Sources panel ──────────────────────────────────────────────
                if show_sources and sources:
                    with st.expander(
                        f"📚 Sources ({len(sources)} documents)",
                        expanded=False,
                    ):
                        for i, chunk in enumerate(sources, 1):
                            st.markdown(f"""
<div class="source-citation">
    <strong>[{i}]</strong>
    <span class="score-badge">Similarity: {chunk.score:.2f}</span><br/>
    <em>Source: {chunk.document.metadata.get('source', 'Unknown')}</em>
    {f"| Topic: {chunk.document.metadata.get('topic', '')}" if chunk.document.metadata.get('topic') else ""}
    <br/><br/>
    {chunk.document.content[:400]}...
</div>
""", unsafe_allow_html=True)

                st.caption(f"Using {pipeline.generator.model_name if pipeline.generator else 'model'}")

            except Exception as e:
                full_answer = "Error generating response"
                sources = []
                st.error(f"Error generating response: {str(e)}")
                st.info("Please check the pipeline logs or try again with a different query.")

    # Add assistant message to history
    assistant_message = {
        "role": "assistant",
        "content": full_answer if 'full_answer' in locals() else "Error generating response",
    }
    if 'sources' in locals() and sources:
        assistant_message["sources"] = sources

    st.session_state.messages.append(assistant_message)


# ============================================================================
# Footer
# ============================================================================

st.divider()
st.markdown("""
<small style="color: #666;">
    Built with ❤️ using Streamlit, FAISS, and Mistral LLM |
    Data: MedQuAD and medical literature |
    Always consult healthcare professionals for medical advice
</small>
""", unsafe_allow_html=True)
