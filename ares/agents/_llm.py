from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

try:  # pragma: no cover - optional dependency
    from anthropic import Anthropic
except ImportError:  # pragma: no cover - optional dependency
    Anthropic = None

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

_LLM_MAX_RETRIES = 3
_LLM_RETRY_BACKOFF = (1.0, 3.0, 6.0)


@dataclass(slots=True)
class LLMAdapter:
    api_key: str = ""
    model: str = ""
    provider: str = "anthropic"
    client: object | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.provider = (self.provider or "anthropic").lower()
        self.client = self._build_client()

    def _build_client(self):
        if not self.api_key:
            return None
        if self.provider == "openai":
            if OpenAI is None:
                return None
            return OpenAI(api_key=self.api_key)
        if Anthropic is None:
            return None
        return Anthropic(api_key=self.api_key)

    @property
    def available(self) -> bool:
        return self.client is not None and bool(self.model)

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 2000, temperature: float = 0.0) -> str:
        if not self.available:
            return ""
        last_error: Exception | None = None
        for attempt in range(_LLM_MAX_RETRIES):
            try:
                return self._call_api(system_prompt, user_prompt, max_tokens, temperature=temperature)
            except Exception as exc:
                last_error = exc
                is_retryable = self._is_retryable(exc)
                if not is_retryable or attempt == _LLM_MAX_RETRIES - 1:
                    print(
                        f"[llm] {self.model} call failed after {attempt + 1} attempt(s): {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return ""
                delay = _LLM_RETRY_BACKOFF[min(attempt, len(_LLM_RETRY_BACKOFF) - 1)]
                print(
                    f"[llm] {self.model} retryable error (attempt {attempt + 1}): {exc}; retrying in {delay:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        return ""

    def _call_api(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float = 0.0) -> str:
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            message = response.choices[0].message.content or ""
            if isinstance(message, list):
                return "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in message
                )
            return str(message)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )
        content = getattr(response, "content", [])
        parts = []
        for item in content:
            parts.append(getattr(item, "text", ""))
        return "".join(parts)

    def complete_batch(
        self,
        requests: list[tuple[str, str, int]],
        poll_interval: int = 10,
        timeout: int = 3600,
    ) -> list[str]:
        """Submit requests via Anthropic Batch API for 50% cost savings.

        Each request is ``(system_prompt, user_prompt, max_tokens)``.
        Returns response strings in the same order.  Requires Anthropic provider.
        """
        if not self.available or self.provider != "anthropic":
            return [""] * len(requests)
        if not requests:
            return []

        batch_requests = []
        for i, (system_prompt, user_prompt, max_tokens) in enumerate(requests):
            batch_requests.append({
                "custom_id": f"req_{i}",
                "params": {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "system": [
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    "messages": [{"role": "user", "content": user_prompt}],
                },
            })

        try:
            batch = self.client.messages.batches.create(requests=batch_requests)
        except Exception as exc:
            print(f"[llm] batch creation failed: {exc}", file=sys.stderr, flush=True)
            return [""] * len(requests)

        batch_id = batch.id
        print(
            f"[llm] batch {batch_id} submitted ({len(requests)} requests)",
            file=sys.stderr, flush=True,
        )

        elapsed = 0
        while elapsed < timeout:
            try:
                status = self.client.messages.batches.retrieve(batch_id)
            except Exception as exc:
                print(f"[llm] batch poll error: {exc}", file=sys.stderr, flush=True)
                time.sleep(poll_interval)
                elapsed += poll_interval
                continue
            if status.processing_status == "ended":
                break
            time.sleep(poll_interval)
            elapsed += poll_interval
        else:
            print(
                f"[llm] batch {batch_id} timed out after {timeout}s",
                file=sys.stderr, flush=True,
            )
            return [""] * len(requests)

        results_map: dict[str, str] = {}
        for entry in self.client.messages.batches.results(batch_id):
            if entry.result.type == "succeeded":
                content = entry.result.message.content
                text = "".join(getattr(item, "text", "") for item in content)
                results_map[entry.custom_id] = text

        return [results_map.get(f"req_{i}", "") for i in range(len(requests))]

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        exc_type = type(exc).__name__
        if exc_type in {"RateLimitError", "APIConnectionError", "InternalServerError", "APITimeoutError"}:
            return True
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        if isinstance(status, int) and status in {429, 500, 502, 503, 529}:
            return True
        msg = str(exc).lower()
        if "rate limit" in msg or "overloaded" in msg or "timeout" in msg:
            return True
        return False
