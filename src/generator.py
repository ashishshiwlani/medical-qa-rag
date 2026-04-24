"""
LLM text generation for RAG answers.

Streaming:
  Both generators now expose generate_stream(query, context) → Iterator[str].

  DemoGenerator (FLAN-T5):  Uses HuggingFace TextIteratorStreamer.
    Inference runs in a background thread; the main thread yields decoded
    tokens as they arrive.  This gives real word-by-word streaming even on CPU.

  LLMGenerator (Mistral-7B):  Same pattern — TextIteratorStreamer wired to
    the AutoModelForCausalLM.generate() call in a daemon thread.

  Streamlit integration (in app/streamlit_app.py):
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_text = ""
        for token in pipeline.generate_stream(query, context):
            full_text += token
            response_placeholder.markdown(full_text + "▌")
        response_placeholder.markdown(full_text)


Handles:
  - System prompt design for medical Q&A (grounding, citation, safety)
  - Loading and configuring LLM models
  - Prompt formatting for specific model architectures
  - Optional 4-bit quantization for memory efficiency

Key concepts:
  - System prompt: critical for controlling LLM behavior in RAG
    Must instruct model to: cite sources, admit knowledge limits, avoid diagnoses
  - 4-bit quantization: reduces 14GB Mistral-7B to ~4GB with minimal quality loss
    Uses BitsAndBytesConfig for automatic quantization
  - Instruction templates: Different models have different formats
    (e.g., Mistral uses [INST] ... [/INST] format)
"""

import os
import threading
from dataclasses import dataclass
from typing import Optional, Iterator

# torch and transformers are only needed inside _load_model().
# Keeping them out of the module-level scope means importing SYSTEM_PROMPT
# or GenerationResult doesn't require a GPU environment to be set up.
# Any function that actually loads a model will hit a clear ImportError
# at that point if the libraries are missing.


# ============================================================================
# SYSTEM PROMPT: Core RAG instructions
# ============================================================================

SYSTEM_PROMPT = """You are a helpful medical information assistant. Your role is to answer questions about medical topics based on the provided context from medical literature.

CRITICAL INSTRUCTIONS:
1. Answer ONLY based on the provided context. Do not use outside knowledge.
2. If the context doesn't contain relevant information, say: "I don't have information about that in the available medical literature."
3. Always cite your sources using the format: "According to [Source Name], ..."
4. NEVER provide definitive medical diagnoses. Use phrases like "may indicate", "can be associated with", "are common symptoms of".
5. ALWAYS recommend consulting a healthcare professional for medical advice or diagnosis.
6. If a question asks about symptoms or conditions, provide factual information but emphasize: "This is educational information only. Please consult a doctor."
7. Do not make up statistics, percentages, or specific medical claims. Only state what's in the context.
8. Be concise and clear. Use bullet points for lists when appropriate.

TONE: Professional, accurate, safety-conscious, and helpful."""


@dataclass
class GenerationResult:
    """
    Result from LLM generation.

    Attributes:
        answer: Generated text
        model_name: Name of model used
        prompt_tokens: Estimated tokens in prompt (for cost/performance tracking)
    """
    answer: str
    model_name: str
    prompt_tokens: int


class LLMGenerator:
    """
    Generate answers using a large language model.

    Supports Mistral-7B-Instruct with optional 4-bit quantization.

    Attributes:
        model_name: HuggingFace model identifier
        tokenizer: Loaded tokenizer
        model: Loaded model
        pipeline: Text generation pipeline
        max_new_tokens: Max tokens to generate per call
        device: GPU/CPU device
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        use_4bit: bool = False
    ) -> None:
        """
        Initialize LLM generator.

        Args:
            model_name: HuggingFace model ID
                Default: Mistral-7B-Instruct (14GB, good quality-speed trade-off)
            device: Device to load on ('cuda', 'cpu')
                If None, auto-detects
            max_new_tokens: Maximum tokens to generate per query
            use_4bit: Whether to use 4-bit quantization
                Reduces VRAM usage from 14GB to ~4GB with minimal quality loss

        Raises:
            RuntimeError: If model loading fails
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        # Defer torch import so __init__ doesn't require torch at construction time
        if device is None:
            try:
                import torch as _torch
                device = "cuda" if _torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device

        print(f"Initializing LLM Generator: {model_name}")
        print(f"Device: {self.device}, 4-bit: {use_4bit}")

        self.tokenizer = None
        self.model = None
        self.pipeline = None

        self._load_model(use_4bit=use_4bit)

    def _load_model(self, use_4bit: bool = False) -> None:
        """
        Load model and tokenizer from HuggingFace.

        Args:
            use_4bit: Whether to apply 4-bit quantization
                Requires bitsandbytes library
        """
        # Lazy imports — only needed at model-load time
        try:
            import torch
            from transformers import (
                AutoTokenizer, AutoModelForCausalLM,
                TextGenerationPipeline, BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required for LLMGenerator. "
                "Install with: pip install torch transformers"
            ) from exc

        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Configure 4-bit quantization if requested
        # Why 4-bit: reduces memory 4x with <1% quality loss
        # Uses NF4 (normal float 4-bit) from bitsandbytes
        quantization_config = None
        if use_4bit:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                print("Using 4-bit quantization (NF4)")
            except Exception as e:
                print(f"Warning: 4-bit quantization failed ({e}), using full precision")
                quantization_config = None

        # Load model
        print("Loading model (this may take a minute)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto" if self.device == "cuda" else self.device,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        # Create text generation pipeline
        # Pipeline handles tokenization, generation, and decoding
        self.pipeline = TextGenerationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else self.device,
        )

        print(f"✓ Loaded {self.model_name}")

    def _build_prompt(self, query: str, context: str) -> str:
        """
        Format query and context into Mistral instruction format.

        Mistral instruction template: [INST] user_message [/INST] assistant_response

        Args:
            query: User question
            context: Retrieved context from RAG

        Returns:
            Formatted prompt string
        """
        # Combine system prompt, context, and user query
        prompt = f"""{SYSTEM_PROMPT}

## Medical Context
{context}

[INST] {query} [/INST]"""

        return prompt

    def generate(self, query: str, context: str) -> GenerationResult:
        """
        Generate an answer given a query and context.

        Args:
            query: User question
            context: Retrieved medical context from RAG

        Returns:
            GenerationResult with answer and metadata

        Raises:
            RuntimeError: If generation fails
        """
        prompt = self._build_prompt(query, context)

        # Estimate token count (rough: ~4 chars per token)
        prompt_tokens = len(prompt) // 4

        try:
            # Generate response
            outputs = self.pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.3,  # Lower = more focused/deterministic
                top_p=0.9,        # Nucleus sampling for diversity
                return_full_text=False,  # Only return generated text, not prompt
            )

            # Extract generated text
            answer = outputs[0]["generated_text"].strip()

            return GenerationResult(
                answer=answer,
                model_name=self.model_name,
                prompt_tokens=prompt_tokens
            )
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")

    def generate_stream(self, query: str, context: str) -> Iterator[str]:
        """
        Stream answer tokens one by one using TextIteratorStreamer.

        Runs model.generate() in a background daemon thread and yields decoded
        tokens from the main thread as they become available.

        Args:
            query:   User question.
            context: Retrieved RAG context string.

        Yields:
            Decoded token strings (may include whitespace / punctuation).

        Example:
            >>> for token in generator.generate_stream("What is diabetes?", ctx):
            ...     print(token, end="", flush=True)
        """
        try:
            import torch
            from transformers import TextIteratorStreamer
        except ImportError:
            # Fall back to yielding the full answer as one chunk
            result = self.generate(query, context)
            yield result.answer
            return

        prompt = self._build_prompt(query, context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {
            **inputs,
            "max_new_tokens":   self.max_new_tokens,
            "do_sample":        True,
            "temperature":      0.3,
            "top_p":            0.9,
            "streamer":         streamer,
        }

        # Run generation in a daemon thread so we can yield from the main thread
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        for token_text in streamer:
            yield token_text

        thread.join()


class DemoGenerator:
    """
    Lightweight generator using FLAN-T5-Base for demo/CPU mode.

    FLAN-T5-Base:
      - 250M parameters vs 7B for Mistral
      - ~1GB vs 14GB memory requirement
      - Runs on CPU in reasonable time
      - Lower quality but sufficient for demos

    Attributes:
        model_name: HuggingFace model identifier
        tokenizer: Loaded tokenizer
        model: Loaded model
        pipeline: Text generation pipeline
    """

    def __init__(self, max_new_tokens: int = 256) -> None:
        """
        Initialize demo generator with FLAN-T5-Base.

        Args:
            max_new_tokens: Max tokens to generate
        """
        self.model_name = "google/flan-t5-base"
        self.max_new_tokens = max_new_tokens

        print("Loading lightweight demo model: FLAN-T5-Base")

        # Lazy imports — keep module importable without torch installed
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required for DemoGenerator. "
                "Install with: pip install torch transformers"
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # CPU: use float32
        )

        self.pipeline = TextGenerationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
        )

        print("✓ Loaded FLAN-T5-Base (demo mode)")

    def generate(self, query: str, context: str) -> GenerationResult:
        """
        Generate answer using FLAN-T5.

        Args:
            query: User question
            context: Retrieved context

        Returns:
            GenerationResult with answer
        """
        # Simpler prompt for FLAN-T5 (not instruction-tuned like Mistral)
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )

        prompt_tokens = len(prompt) // 4

        try:
            outputs = self.pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.3,
                return_full_text=False,
            )

            answer = outputs[0]["generated_text"].strip()

            return GenerationResult(
                answer=answer,
                model_name=self.model_name,
                prompt_tokens=prompt_tokens
            )
        except Exception as e:
            # Fallback: return a safe error message
            return GenerationResult(
                answer=(
                    "I encountered an error while generating a response. "
                    "Please try a different query or check the logs."
                ),
                model_name=self.model_name,
                prompt_tokens=prompt_tokens
            )

    def generate_stream(self, query: str, context: str) -> Iterator[str]:
        """
        Stream FLAN-T5 answer tokens using TextIteratorStreamer.

        Uses a background thread exactly like LLMGenerator.generate_stream().
        Falls back to yielding the full answer at once if streaming is unavailable.

        Args:
            query:   User question.
            context: Retrieved RAG context.

        Yields:
            Decoded token strings.
        """
        try:
            import torch
            from transformers import TextIteratorStreamer
        except ImportError:
            result = self.generate(query, context)
            yield result.answer
            return

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample":      True,
            "temperature":    0.3,
            "streamer":       streamer,
        }

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        for token_text in streamer:
            yield token_text

        thread.join()
