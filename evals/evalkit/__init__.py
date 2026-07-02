"""evalkit — Phase 6 evaluation harness for the Qwen3.6-27B agentic assistant.

Design rules:
    * Every grade is VERIFIABLE — executed code, checked environment state,
      parsed tool-call JSON, numeric tolerance, mechanical constraint checks.
      No LLM judges. This is what lets the suite double as an RLVR reward
      environment for a future v2 (the Phase 1 plan).
    * The model is reached through any OpenAI-compatible chat endpoint
      (vLLM, llama.cpp server, Ollama), so identical runs work against the
      base model and the fine-tune.
    * Like trainkit, everything here imports without GPU frameworks; the
      full test suite runs on a CPU-only box.
"""

__version__ = "0.1.0"
