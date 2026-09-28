"""OpenAI-compatible chat client with retries, request pacing and an on-disk response cache."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

from .config import PROVIDERS


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, provider: str = "gemini", model: str | None = None, api_key: str | None = None,
                 min_interval: float = 0.0, max_retries: int = 6, timeout: float = 90.0,
                 cache_dir: Path | None = None):
        from openai import OpenAI

        preset = PROVIDERS[provider]
        key = api_key or os.getenv(preset["key_env"], "")
        if not key:
            raise LLMError(f"set {preset['key_env']} (or pass api_key)")
        self.model = model or preset["model"]
        self.client = OpenAI(api_key=key, base_url=preset["base_url"], timeout=timeout, max_retries=0)
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0           # requests actually sent
        self.cache_hits = 0
        self._last = 0.0

    # -- internals ---------------------------------------------------------
    def _key(self, payload: dict) -> Path:
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _send(self, **kwargs):
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

        delay = 4.0
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                self.calls += 1
                return self.client.chat.completions.create(model=self.model, **kwargs)
            except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as exc:
                if attempt == self.max_retries:
                    raise LLMError(f"{type(exc).__name__} after {attempt + 1} attempts") from None
                time.sleep(delay + random.random())
                delay = min(delay * 2, 60.0)
        raise LLMError("unreachable")

    # -- public ------------------------------------------------------------
    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 900,
             json_mode: bool = False) -> str:
        payload = {"model": self.model, "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens, "json": json_mode}
        path = self._key(payload) if self.cache_dir else None
        if path and path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text())["text"]
        kwargs = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        text = (self._send(**kwargs).choices[0].message.content or "").strip()
        if path:
            path.write_text(json.dumps({"text": text}))
        return text

    def chat_tools(self, messages: list[dict], tools: list[dict], max_tokens: int = 900):
        """One tool-calling turn; returns the raw assistant message."""
        return self._send(messages=messages, tools=tools, tool_choice="auto", temperature=0.0,
                          max_tokens=max_tokens).choices[0].message
