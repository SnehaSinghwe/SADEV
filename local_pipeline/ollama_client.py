"""
local_pipeline/ollama_client.py
Thin wrapper around the Ollama REST API.
Supports both streaming and non-streaming calls to Gemma 2.

Ollama must be running: `ollama serve`
Model must be pulled:   `ollama pull gemma2` or `ollama pull gemma2:9b`

The client handles:
  - Chat format (system + messages array)
  - Timeout and connection error graceful fallback
  - Optional streaming with a callback
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Callable, Optional


class OllamaClient:
    """
    Synchronous Ollama client.
    Uses urllib (no extra deps) so the whole prototype works with just
    the packages already in requirements.txt.
    """

    def __init__(
        self,
        model: str = "gemma2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 400,
        timeout: int = 60,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(
        self,
        system_prompt: str,
        messages: list[dict],
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Send a chat request to Ollama. Returns the complete response text.

        messages: list of {"role": "user"|"assistant", "content": str}
        stream_callback: if provided, called with each token chunk as it arrives.
        """
        url = f"{self.base_url}/api/chat"

        # Build Ollama messages array with system prepended
        ollama_messages = [{"role": "system", "content": system_prompt}] + messages

        payload = json.dumps({
            "model": self.model,
            "messages": ollama_messages,
            "stream": stream_callback is not None,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "stop": ["</s>", "[INST]", "[/INST]"],
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if stream_callback is not None:
                    return self._handle_stream(resp, stream_callback)
                else:
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw)
                    return data.get("message", {}).get("content", "").strip()

        except urllib.error.URLError as e:
            raise OllamaConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                f"Is `ollama serve` running? Error: {e}"
            ) from e
        except Exception as e:
            raise OllamaGenerationError(f"Generation failed: {e}") from e

    def _handle_stream(
        self,
        response,
        callback: Callable[[str], None],
    ) -> str:
        """Read streaming NDJSON from Ollama and collect full text."""
        full_text = []
        for line in response:
            line = line.decode("utf-8").strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    callback(token)
                    full_text.append(token)
                if chunk.get("done"):
                    break
            except json.JSONDecodeError:
                continue
        return "".join(full_text).strip()

    def check_health(self) -> dict:
        """Check if Ollama is running and the model is available."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                model_available = any(
                    m.startswith(self.model) for m in models
                )
                return {
                    "ollama_running": True,
                    "model_available": model_available,
                    "available_models": models,
                    "requested_model": self.model,
                }
        except Exception as e:
            return {
                "ollama_running": False,
                "model_available": False,
                "error": str(e),
            }


class OllamaConnectionError(RuntimeError):
    """Raised when Ollama is unreachable."""

class OllamaGenerationError(RuntimeError):
    """Raised when generation fails inside Ollama."""
