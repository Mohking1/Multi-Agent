import ast
import json
import logging
import os
import re
import time
from typing import Any, Type, TypeVar

import httpx
from pydantic import BaseModel

from workos_engine.debug import get_tracer

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _clean_response(text: str) -> str:
    """Strips <think>...</think> reasoning blocks, orphan think tags, and leading/trailing whitespace."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"^(?:\{\"?\s*)?</think>\s*", "", cleaned)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def parse_json_strictly(text: str) -> Any:
    """Strictly parses JSON from LLM output without fragile regex heuristics."""
    if not text or not isinstance(text, str):
        return None

    cleaned = _clean_response(text)

    # 1. Direct JSON parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 2. Markdown fenced block
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except Exception:
            pass

    # 3. Outermost JSON object or array slice
    start_obj = cleaned.find("{")
    start_arr = cleaned.find("[")

    if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
        end_obj = cleaned.rfind("}")
        if end_obj > start_obj:
            try:
                return json.loads(cleaned[start_obj : end_obj + 1])
            except Exception:
                pass

    if start_arr != -1:
        end_arr = cleaned.rfind("]")
        if end_arr > start_arr:
            try:
                return json.loads(cleaned[start_arr : end_arr + 1])
            except Exception:
                pass

    return None


def repair_json_string(text: str) -> Any:
    """
    Parses or repairs mildly malformed JSON strings from LLMs.
    Handles:
    1. Valid JSON & Markdown fenced blocks
    2. Trailing commas (e.g. `{"a": 1,}`)
    3. Missing commas between objects or key-values (e.g. `{"a": 1 "b": 2}`)
    4. Truncated JSON (unclosed strings, braces, or brackets)
    """
    if not text or not isinstance(text, str):
        return None

    cleaned = _clean_response(text)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        res = repair_json_string(fence_match.group(1).strip())
        if res is not None:
            return res

    start_obj = cleaned.find("{")
    start_arr = cleaned.find("[")
    if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
        candidate = cleaned[start_obj:]
    elif start_arr != -1:
        candidate = cleaned[start_arr:]
    else:
        candidate = cleaned

    try:
        return json.loads(candidate)
    except Exception:
        pass

    # Clean trailing commas
    fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(fixed)
    except Exception:
        pass

    # Insert missing commas: e.g. 1 "key" or "val" "key" or } {
    fixed_commas = re.sub(r'([}\]0-9"truefalseFalseTrueNone])\s+("|\{)', r"\1, \2", fixed)
    fixed_commas = re.sub(r",\s*([}\]])", r"\1", fixed_commas)
    try:
        return json.loads(fixed_commas)
    except Exception:
        pass

    # Reduce accidental duplicate closing braces if closing braces exceed opening braces
    for cand in (fixed_commas, fixed, candidate):
        if cand.count("}") > cand.count("{"):
            diff = cand.count("}") - cand.count("{")
            reduced = re.sub(r"\}\s*\}", "}", cand, count=diff)
            try:
                return json.loads(reduced)
            except Exception:
                pass

    # Truncation healing: close open strings and balance unclosed braces/brackets
    for attempt in [fixed_commas, fixed, candidate]:
        in_s = False
        esc = False
        for ch in attempt:
            if ch == "\\" and not esc:
                esc = True
                continue
            if ch == '"' and not esc:
                in_s = not in_s
            esc = False
        healed = attempt + ('"' if in_s else "")
        healed = re.sub(r"[,:\s]+$", "", healed)

        stack = []
        in_s = False
        esc = False
        for c in healed:
            if c == "\\" and not esc:
                esc = True
                continue
            if c == '"' and not esc:
                in_s = not in_s
            elif not in_s:
                if c in "{[":
                    stack.append("}" if c == "{" else "]")
                elif c in "}]" and stack and stack[-1] == c:
                    stack.pop()
            esc = False
        closing = "".join(reversed(stack))
        try:
            return json.loads(healed + closing)
        except Exception:
            pass

    return None



def extract_and_parse_json(text: str, default: Any = None, session_id: str | None = None) -> Any:
    """
    Robust JSON extraction supporting:
    - Raw JSON string
    - Markdown fenced blocks (```json ... ```)
    - Embedded JSON objects {...} or arrays [...]
    - Multi-strategy self-repair for LLM syntax imperfections
    """
    if not text or not isinstance(text, str):
        return default

    cleaned = _clean_response(text)
    tracer = get_tracer()

    # 1. Check markdown code fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
        repaired = repair_json_string(candidate)
        if repaired is not None:
            tracer.log_json_parse(
                "llm_client",
                candidate,
                success=True,
                parsed_data=repaired,
                strategy="markdown_fence",
                session_id=session_id,
            )
            return repaired

    # 2. Try whole cleaned string
    repaired = repair_json_string(cleaned)
    if repaired is not None:
        tracer.log_json_parse(
            "llm_client",
            cleaned,
            success=True,
            parsed_data=repaired,
            strategy="whole_string",
            session_id=session_id,
        )
        return repaired

    # 3. Locate outermost object {...} or array [...]
    start_obj = cleaned.find("{")
    start_arr = cleaned.find("[")

    if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
        end_obj = cleaned.rfind("}")
        candidate = cleaned[start_obj : end_obj + 1] if end_obj > start_obj else cleaned[start_obj:]
        repaired = repair_json_string(candidate)
        if repaired is not None:
            tracer.log_json_parse(
                "llm_client",
                candidate,
                success=True,
                parsed_data=repaired,
                strategy="outer_object",
                session_id=session_id,
            )
            return repaired

    if start_arr != -1:
        end_arr = cleaned.rfind("]")
        candidate = cleaned[start_arr : end_arr + 1] if end_arr > start_arr else cleaned[start_arr:]
        repaired = repair_json_string(candidate)
        if repaired is not None:
            tracer.log_json_parse(
                "llm_client",
                candidate,
                success=True,
                parsed_data=repaired,
                strategy="outer_array",
                session_id=session_id,
            )
            return repaired

    tracer.log_json_parse(
        "llm_client",
        cleaned,
        success=False,
        error="All JSON repair strategies exhausted",
        session_id=session_id,
    )
    return default


class OllamaClient:
    """
    Lightweight, native client communicating directly with Ollama's REST API.
    Supports generation (with optional JSON format mode), chat, embeddings, and model listing.
    Uses standardized executive hyperparameters (temp 0.6, top_p 0.95, repeat_penalty 1.1, num_ctx 12288).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "refinedneuro/refinedtoolcallv5-3b",
        embedding_model: str = "bge-m3:latest",
        timeout: float = 360.0,
        temperature: float = 0.6,
        top_p: float = 0.95,
        repeat_penalty: float = 1.1,
        num_ctx: int = 6144,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.embedding_model = embedding_model
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.num_ctx = num_ctx

    def _build_options(
        self,
        temperature: float | None = None,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
    ) -> dict[str, Any]:
        return {
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p if top_p is None else top_p,
            "repeat_penalty": self.repeat_penalty if repeat_penalty is None else repeat_penalty,
            "num_ctx": self.num_ctx if num_ctx is None else num_ctx,
            "num_predict": 2048 if num_predict is None else num_predict,
        }

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        format: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
    ) -> str:
        """
        Generate text completion from Ollama (/api/generate).
        Uses standardized executive hyperparameters (temp 0.6, top_p 0.95, repeat_penalty 1.1, num_ctx 6144).
        """
        target_model = model or self.default_model
        payload: dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": self._build_options(
                temperature, top_p, repeat_penalty, num_ctx, num_predict
            ),
        }
        if system:
            payload["system"] = system
        if format:
            payload["format"] = format

        tracer = get_tracer()
        tracer.log_prompt(
            component="OllamaClient",
            prompt=prompt,
            system_prompt=system,
            model=target_model,
            format_type=str(format) if format else None,
            temperature=temperature if temperature is not None else self.temperature,
        )

        start_t = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(f"{self.base_url}/api/generate", json=payload)
                res.raise_for_status()
                data = res.json()
                clean_text = _clean_response(data.get("response", ""))
                duration_ms = (time.perf_counter() - start_t) * 1000.0
                tracer.log_llm_response(
                    component="OllamaClient",
                    response_text=clean_text,
                    model=target_model,
                    duration_ms=duration_ms,
                )
                return clean_text
        except Exception as e:
            duration_ms = (time.perf_counter() - start_t) * 1000.0
            tracer.log_llm_response(
                component="OllamaClient",
                response_text="",
                model=target_model,
                duration_ms=duration_ms,
                error=str(e),
            )
            logger.warning(f"Ollama generate failed on {self.base_url} ({payload['model']}): {e}")
            raise

    async def generate_async(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        format: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
    ) -> str:
        """Asynchronous generation from Ollama."""
        target_model = model or self.default_model
        payload: dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": self._build_options(
                temperature, top_p, repeat_penalty, num_ctx, num_predict
            ),
        }
        if system:
            payload["system"] = system
        if format:
            payload["format"] = format

        tracer = get_tracer()
        tracer.log_prompt(
            component="OllamaClient.async",
            prompt=prompt,
            system_prompt=system,
            model=target_model,
            format_type=str(format) if format else None,
            temperature=temperature if temperature is not None else self.temperature,
        )

        start_t = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(f"{self.base_url}/api/generate", json=payload)
                res.raise_for_status()
                data = res.json()
                clean_text = _clean_response(data.get("response", ""))
                duration_ms = (time.perf_counter() - start_t) * 1000.0
                tracer.log_llm_response(
                    component="OllamaClient.async",
                    response_text=clean_text,
                    model=target_model,
                    duration_ms=duration_ms,
                )
                return clean_text
        except Exception as e:
            duration_ms = (time.perf_counter() - start_t) * 1000.0
            tracer.log_llm_response(
                component="OllamaClient.async",
                response_text="",
                model=target_model,
                duration_ms=duration_ms,
                error=str(e),
            )
            logger.warning(f"Ollama generate_async failed on {self.base_url}: {e}")
            raise

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        model: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
        num_ctx: int | None = None,
    ) -> T:
        """Generates structured output constrained by a Pydantic schema using Ollama's JSON schema grammar."""
        json_schema = schema.model_json_schema()
        raw = self.generate(
            prompt=prompt,
            model=model,
            system=system,
            format=json_schema,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            num_ctx=num_ctx,
        )
        return schema.model_validate_json(raw)

    async def generate_structured_async(
        self,
        prompt: str,
        schema: Type[T],
        model: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
        num_ctx: int | None = None,
    ) -> T:
        """Asynchronously generates structured output constrained by a Pydantic schema."""
        json_schema = schema.model_json_schema()
        raw = await self.generate_async(
            prompt=prompt,
            model=model,
            system=system,
            format=json_schema,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            num_ctx=num_ctx,
        )
        return schema.model_validate_json(raw)

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        format: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
        num_ctx: int | None = None,
    ) -> str:
        """Chat completion from Ollama (/api/chat)."""
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": False,
        }
        if format:
            payload["format"] = format

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(f"{self.base_url}/api/chat", json=payload)
                res.raise_for_status()
                data = res.json()
                return _clean_response(data.get("message", {}).get("content", ""))
        except Exception as e:
            logger.warning(f"Ollama chat failed on {self.base_url}: {e}")
            raise

    def embed(
        self,
        input_text: str | list[str],
        model: str | None = None,
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


class GeminiClient:
    """
    Cloud intelligence client communicating with Google Gemini API via google-genai SDK.
    Provides identical interface to OllamaClient (generate, generate_async, chat, embed, list_models).
    """

    def __init__(
        self,
        api_key: str,
        default_model: str = "gemini-3.6-flash",
        embedding_model: str = "gemini-embedding-001",
        timeout: float = 60.0,
        temperature: float = 0.6,
        top_p: float = 0.95,
        repeat_penalty: float = 1.1,
        num_ctx: int = 12288,
    ):
        if genai is None:
            raise ImportError("google-genai package is not installed. Please install google-genai.")
        self.api_key = api_key
        self.default_model = default_model
        self.model = default_model
        self.embedding_model = embedding_model
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.num_ctx = num_ctx
        self.client = genai.Client(api_key=self.api_key)

    FALLBACK_MODELS: list[str] = [
        "gemini-flash-latest",
        "gemini-3.5-flash",
        "gemini-3.7-flash",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
    ]

    def _resolve_model(self, model: str | None) -> str:
        if not model:
            return self.default_model
        m_lower = model.lower()
        if "gemini" in m_lower or "gemma" in m_lower or m_lower.startswith("models/"):
            return model
        return self.default_model

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        format: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
    ) -> str:
        target_model = self._resolve_model(model)
        gen_config = genai_types.GenerateContentConfig(
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p if top_p is None else top_p,
        )
        if system:
            gen_config.system_instruction = system
        if format in ("json", "application/json"):
            gen_config.response_mime_type = "application/json"
        if num_predict:
            gen_config.max_output_tokens = num_predict

        tracer = get_tracer()
        models_to_try = [target_model] + [m for m in self.FALLBACK_MODELS if m != target_model]
        last_err: Exception | None = None

        for cur_model in models_to_try:
            tracer.log_prompt(
                component="GeminiClient",
                prompt=prompt,
                system_prompt=system,
                model=cur_model,
                format_type=format,
                temperature=temperature if temperature is not None else self.temperature,
            )
            start_t = time.perf_counter()
            try:
                res = self.client.models.generate_content(
                    model=cur_model,
                    contents=prompt,
                    config=gen_config,
                )
                clean_text = _clean_response(res.text or "")
                duration_ms = (time.perf_counter() - start_t) * 1000.0
                tracer.log_llm_response(
                    component="GeminiClient",
                    response_text=clean_text,
                    model=cur_model,
                    duration_ms=duration_ms,
                )
                return clean_text
            except Exception as e:
                last_err = e
                duration_ms = (time.perf_counter() - start_t) * 1000.0
                err_str = str(e).lower()
                if "429" in err_str or "exhausted" in err_str or "quota" in err_str:
                    logger.warning(f"Gemini model '{cur_model}' hit quota limit, failing over: {e}")
                    continue
                tracer.log_llm_response(
                    component="GeminiClient",
                    response_text="",
                    model=cur_model,
                    duration_ms=duration_ms,
                    error=str(e),
                )
                raise

        if last_err:
            raise last_err
        return ""

    async def generate_async(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        format: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
    ) -> str:
        target_model = self._resolve_model(model)
        gen_config = genai_types.GenerateContentConfig(
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p if top_p is None else top_p,
        )
        if system:
            gen_config.system_instruction = system
        if format in ("json", "application/json"):
            gen_config.response_mime_type = "application/json"
        if num_predict:
            gen_config.max_output_tokens = num_predict

        tracer = get_tracer()
        models_to_try = [target_model] + [m for m in self.FALLBACK_MODELS if m != target_model]
        last_err: Exception | None = None

        for cur_model in models_to_try:
            tracer.log_prompt(
                component="GeminiClient.async",
                prompt=prompt,
                system_prompt=system,
                model=cur_model,
                format_type=format,
                temperature=temperature if temperature is not None else self.temperature,
            )
            start_t = time.perf_counter()
            try:
                res = await self.client.aio.models.generate_content(
                    model=cur_model,
                    contents=prompt,
                    config=gen_config,
                )
                clean_text = _clean_response(res.text or "")
                duration_ms = (time.perf_counter() - start_t) * 1000.0
                tracer.log_llm_response(
                    component="GeminiClient.async",
                    response_text=clean_text,
                    model=cur_model,
                    duration_ms=duration_ms,
                )
                return clean_text
            except Exception as e:
                last_err = e
                duration_ms = (time.perf_counter() - start_t) * 1000.0
                err_str = str(e).lower()
                if "429" in err_str or "exhausted" in err_str or "quota" in err_str:
                    logger.warning(f"Gemini model '{cur_model}' hit quota limit, failing over: {e}")
                    continue
                tracer.log_llm_response(
                    component="GeminiClient.async",
                    response_text="",
                    model=cur_model,
                    duration_ms=duration_ms,
                    error=str(e),
                )
                raise

        if last_err:
            raise last_err
        return ""

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        format: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
        num_ctx: int | None = None,
    ) -> str:
        target_model = self._resolve_model(model)
        gen_config = genai_types.GenerateContentConfig(
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p if top_p is None else top_p,
        )
        if format in ("json", "application/json"):
            gen_config.response_mime_type = "application/json"

        system_instruction = None
        contents = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system_instruction = content
            else:
                gemini_role = "model" if role in ("assistant", "model") else "user"
                contents.append(
                    genai_types.Content(
                        role=gemini_role,
                        parts=[genai_types.Part.from_text(text=content)],
                    )
                )

        if system_instruction:
            gen_config.system_instruction = system_instruction

        models_to_try = [target_model] + [m for m in self.FALLBACK_MODELS if m != target_model]
        last_err: Exception | None = None

        for cur_model in models_to_try:
            try:
                res = self.client.models.generate_content(
                    model=cur_model,
                    contents=contents,
                    config=gen_config,
                )
                return _clean_response(res.text or "")
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if "429" in err_str or "exhausted" in err_str or "quota" in err_str:
                    logger.warning(f"Gemini model '{cur_model}' hit quota limit in chat, failing over: {e}")
                    continue
                logger.warning(f"Gemini chat failed ({cur_model}): {e}")
                raise

        if last_err:
            raise last_err
        return ""

    def embed(
        self,
        input_text: str | list[str],
        model: str | None = None,
    ) -> list[list[float]]:
        embed_model = model or self.embedding_model or "gemini-embedding-001"
        texts = [input_text] if isinstance(input_text, str) else input_text
        results = []
        try:
            for t in texts:
                res = self.client.models.embed_content(
                    model=embed_model,
                    contents=t,
                )
                if res.embeddings:
                    results.append(list(res.embeddings[0].values))
            if len(results) == len(texts):
                return results
        except Exception as e:
            logger.debug(f"Gemini embedding failed ({embed_model}): {e}")
        return []

    def list_models(self) -> list[str]:
        try:
            return [m.name for m in self.client.models.list()]
        except Exception as e:
            logger.debug(f"Could not fetch Gemini models list: {e}")
            return [self.default_model]

    def is_available(self) -> bool:
        return bool(self.api_key)


class MockLLMClient:
    """
    Deterministic, offline mock client for fast, reproducible testing.
    Provides identical interface to OllamaClient with zero network or GPU requirements.
    """

    def __init__(
        self,
        default_response: str = "Mock LLM Response",
        canned_responses: dict[str, Any] | None = None,
        structured_mocks: dict[str, Any] | None = None,
    ):
        self.default_response = default_response
        self.canned_responses = canned_responses or {}
        self.structured_mocks = structured_mocks or {}
        self.default_model = "mock-llm"
        self.embedding_model = "mock-embed"
        self.history: list[dict[str, Any]] = []

    def set_response(self, match_key: str, response: Any) -> None:
        self.canned_responses[match_key] = response

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        format: Any = None,
        **kwargs: Any,
    ) -> str:
        self.history.append({"action": "generate", "prompt": prompt, "system": system, "format": format})
        for key, resp in self.canned_responses.items():
            if key.lower() in prompt.lower() or (system and key.lower() in system.lower()):
                if isinstance(resp, (dict, list)):
                    return json.dumps(resp)
                return str(resp)

        # Generic mock logic for candidate email classification tests
        if ("candidate" in prompt.lower()) and (
            "is_match" in prompt.lower() or "classifications" in prompt.lower()
        ):
            try:
                cand_match = re.search(r"Candidate(?:s|\s+Email):\s*(\[?\s*\{[\s\S]*?\}\s*\]?)", prompt)
                if cand_match:
                    raw_json = cand_match.group(1).strip()
                    parsed_cand = json.loads(raw_json)
                    cands = parsed_cand if isinstance(parsed_cand, list) else [parsed_cand]
                    c = cands[0] if cands else {}
                    from_addr = (c.get("from") or "").lower()
                    subj = (c.get("subject") or "").lower()
                    snip = (c.get("snippet") or "").lower()
                    domain_part = from_addr.split("@")[-1].split(".")[0].title() if "@" in from_addr else "General"
                    # Noise rejection for synthetic test fixtures
                    is_noise = any(w in from_addr or w in subj or w in snip for w in ("sbi", "naukri", "quora", "reddit", "digest", "newsletter"))
                    is_match = not is_noise

                    resp_dict = {
                        "uid": str(c.get("uid", "")),
                        "is_match": is_match,
                        "subcategory": domain_part,
                        "subfolder": domain_part,
                        "reason": "Mock classified",
                    }
                    return json.dumps(resp_dict)
            except Exception as e:
                logger.debug(f"Mock classification error: {e}")

        # Generic mock logic for mail organization spec
        if "mail organization specialist" in prompt.lower() or ("search_terms" in prompt.lower() and "target_folder" in prompt.lower()):
            m_f = re.search(r"(?:folder|label)\s+(?:named|called)\s+['\"]?([^'\",\.\n]+)", prompt, re.IGNORECASE)
            folder_name = m_f.group(1).strip() if m_f else "Organized Mail"
            m_sub = re.search(r"sub\s*folders?\s+(?:based\s+upon|by|named)\s+([^,\n\.]+)", prompt, re.IGNORECASE)
            sub_rule = m_sub.group(1).strip() if m_sub else "none"
            return json.dumps({
                "target_folder": folder_name,
                "subfolder_rule": sub_rule,
                "search_terms": [folder_name.lower()],
                "sender_patterns": [],
                "match_criteria": folder_name.lower(),
            })

        if format in ("json", "application/json") or isinstance(format, dict):
            return "{}"
        return self.default_response

    async def generate_async(self, prompt: str, **kwargs: Any) -> str:
        return self.generate(prompt, **kwargs)

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        **kwargs: Any,
    ) -> T:
        for key, val in self.structured_mocks.items():
            if key.lower() in prompt.lower():
                if isinstance(val, schema):
                    return val
                if isinstance(val, dict):
                    return schema.model_validate(val)
        raw = self.generate(prompt, format=schema.model_json_schema(), **kwargs)
        try:
            return schema.model_validate_json(raw)
        except Exception:
            return schema.model_construct()

    async def generate_structured_async(self, prompt: str, schema: Type[T], **kwargs: Any) -> T:
        return self.generate_structured(prompt, schema, **kwargs)

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        combined = " ".join(m.get("content", "") for m in messages)
        return self.generate(prompt=combined, **kwargs)

    def embed(self, input_text: str | list[str], **kwargs: Any) -> list[list[float]]:
        texts = [input_text] if isinstance(input_text, str) else input_text
        return [[0.01 * (i + 1)] * 128 for i in range(len(texts))]

    def list_models(self) -> list[str]:
        return ["mock-llm", "refinedneuro/refinedtoolcallv5-3b"]

    def is_available(self) -> bool:
        return True


def get_llm_client(config: Any | None = None) -> Any:
    """
    Factory creating the configured LLM client:
    - MockLLMClient if provider is 'mock' or WORKOS_MOCK_LLM=1
    - OllamaClient for local inference (default)
    - GeminiClient if explicitly configured
    """
    if config is None:
        from config import get_config

        config = get_config()

    provider = getattr(config, "llm_provider", "ollama").lower()
    if provider == "mock" or os.getenv("WORKOS_MOCK_LLM", "0") in ("1", "true", "yes"):
        return MockLLMClient()

    gemini_key = (
        getattr(config, "gemini_api_key", "")
        or os.getenv("GEMINI_API_KEY", "")
        or os.getenv("GOOGLE_API_KEY", "")
    )

    if provider == "gemini" and gemini_key and genai is not None:
        return GeminiClient(
            api_key=gemini_key,
            default_model=getattr(config, "gemini_model", "gemini-flash-latest"),
            temperature=getattr(config, "default_temperature", 0.6),
            top_p=getattr(config, "top_p", 0.95),
            repeat_penalty=getattr(config, "repeat_penalty", 1.1),
            num_ctx=getattr(config, "num_ctx", 6144),
        )

    return OllamaClient(
        base_url=getattr(config, "ollama_base_url", "http://localhost:11434"),
        default_model=getattr(config, "model_name", "refinedneuro/refinedtoolcallv5-3b"),
        embedding_model=getattr(config, "embedding_model", "bge-m3:latest"),
        temperature=getattr(config, "default_temperature", 0.6),
        top_p=getattr(config, "top_p", 0.95),
        repeat_penalty=getattr(config, "repeat_penalty", 1.1),
        num_ctx=getattr(config, "num_ctx", 6144),
    )
