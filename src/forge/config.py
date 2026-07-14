"""Central configuration. Every knob is overridable via FORGE_* environment
variables or a .env file, so behaviour can be tuned without code changes."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ForgeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FORGE_", env_file=".env", extra="ignore")

    # LLM backend
    ollama_host: str = "http://localhost:11434"
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

    # Repository scanning
    max_tree_entries: int = 400
    max_summary_chars: int = 12_000

    # Retrieval — embedding model is optional; empty string disables dense
    # retrieval and the engine runs BM25-only (e.g. "nomic-embed-text")
    embedding_model: str = ""
    retrieval_top_k: int = 5
