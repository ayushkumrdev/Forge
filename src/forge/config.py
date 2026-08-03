"""Central configuration. Every knob is overridable via FORGE_* environment
variables or a .env file, so behaviour can be tuned without code changes."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ForgeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FORGE_", env_file=".env", extra="ignore")

    # LLM backend — provider "ollama" (default) or "openai" (any
    # OpenAI-compatible server: LM Studio, llama.cpp, vLLM, OpenRouter, ...)
    provider: str = "ollama"
    ollama_host: str = "http://localhost:11434"
    openai_base_url: str = "http://localhost:1234/v1"
    openai_api_key: str = ""
    model: str = "qwen2.5-coder:7b"
    temperature: float = 0.2
    num_ctx: int = 16384
    request_timeout_s: float = 600.0

    # Agent loop limits (stopping criteria)
    max_agent_steps: int = 25
    max_review_cycles: int = 3
    max_plan_tasks: int = 8

    # Tool behaviour
    command_timeout_s: float = 300.0
    max_tool_output_chars: int = 8_000
    # Parse-verify every write/edit; changes that introduce a syntax error are
    # refused before touching disk (FORGE_SYNTAX_GATE=0 to disable)
    syntax_gate: bool = True

    # Repository scanning
    max_tree_entries: int = 400
    max_summary_chars: int = 12_000

    # Retrieval — embedding model is optional; empty string disables dense
    # retrieval and the engine runs BM25-only (e.g. "nomic-embed-text")
    embedding_model: str = ""
    retrieval_top_k: int = 5

    # GitHub — optional token raises API limits (60/h -> 5000/h) and unlocks
    # private repositories for the github_repo / github_file tools
    github_token: str = ""

    # Chat — history length that triggers LLM compaction of older turns
    chat_compact_threshold: int = 30

    # Two-model brain — optional reasoning model that interprets the user's
    # intent and briefs the coder model before it acts (e.g.
    # "qwen2.5:7b-instruct" thinks, qwen2.5-coder:7b implements).
    # Empty string disables the thinker stage.
    thinker_model: str = ""
