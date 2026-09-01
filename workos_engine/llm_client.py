"""Native Ollama Client for WorkOS."""
import json
import logging
from typing import Any, Optional
import httpx

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Lightweight, native client communicating directly with Ollama's REST API.
    Supports generation (with optional JSON format mode), chat, embeddings, and model listing.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "hf.co/unsloth/SmolLM3-3B-GGUF:UD-Q6_K_XL",
        embedding_model: str = "bge-m3:latest",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.embedding_model = embedding_model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        format: Optional[str] = None,
        temperature: float = 0.1,
    ) -> str:
        """
        Generate text completion from Ollama (/api/generate).
        If format="json", Ollama constrains generation to valid JSON.
        """
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        if format:
            payload["format"] = format

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(f"{self.base_url}/api/generate", json=payload)
                res.raise_for_status()
                data = res.json()
                return data.get("response", "")
        except Exception as e:
            logger.warning(f"Ollama generate failed on {self.base_url} ({payload['model']}): {e}")
            raise

    async def generate_async(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        format: Optional[str] = None,
        temperature: float = 0.1,
    ) -> str:
        """Asynchronous generation from Ollama."""
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        if format:
            payload["format"] = format

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(f"{self.base_url}/api/generate", json=payload)
                res.raise_for_status()
                data = res.json()
                return data.get("response", "")
        except Exception as e:
            logger.warning(f"Ollama generate_async failed on {self.base_url}: {e}")
            raise

    def chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        format: Optional[str] = None,
        temperature: float = 0.1,
    ) -> str:
        """Chat completion from Ollama (/api/chat)."""
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if format:
            payload["format"] = format

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(f"{self.base_url}/api/chat", json=payload)
                res.raise_for_status()
                data = res.json()
                return data.get("message", {}).get("content", "")
        except Exception as e:
            logger.warning(f"Ollama chat failed on {self.base_url}: {e}")
            raise

    def embed(
        self,
        input_text: str | list[str],
        model: Optional[str] = None,
    ) -> list[list[float]]:
        """
        Generate dense vector embeddings from Ollama (/api/embed or /api/embeddings).
        """
        embed_model = model or self.embedding_model or self.default_model
        texts = [input_text] if isinstance(input_text, str) else input_text
        payload = {"model": embed_model, "input": texts}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(f"{self.base_url}/api/embed", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    embeddings = data.get("embeddings", [])
                    if embeddings:
                        return embeddings

                # Fallback to single /api/embeddings endpoint per text
                results = []
                for t in texts:
                    res_single = client.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": embed_model, "prompt": t},
                    )
                    if res_single.status_code == 200:
                        results.append(res_single.json().get("embedding", []))
                    else:
                        break
                if len(results) == len(texts):
                    return results
        except Exception as e:
            logger.debug(f"Ollama embedding failed ({embed_model}): {e}")

        return []

    def list_models(self) -> list[str]:
        """List all available models in local Ollama instance (/api/tags)."""
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.debug(f"Could not fetch Ollama models list: {e}")
        return []

    def is_available(self) -> bool:
        """Check if local Ollama daemon is reachable."""
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

