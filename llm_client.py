"""Provider abstraction for Gemini and NVIDIA NIM calls."""

import logging
import os
import re
import socket
import time

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower().strip()

# Preserve the existing IPv4 preference for provider SDK network calls.
_original_getaddrinfo = socket.getaddrinfo


def _ipv4_getaddrinfo(host, port, family=0, *args, **kwargs):
    return _original_getaddrinfo(host, port, socket.AF_INET, *args, **kwargs)


socket.getaddrinfo = _ipv4_getaddrinfo
_gemini_client = None
_nvidia_client = None


def _get_gemini_client():
    global _gemini_client
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Gemini API key is not configured.")
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _get_nvidia_client():
    global _nvidia_client
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("NVIDIA API key is not configured.")
    if _nvidia_client is None:
        from openai import OpenAI
        _nvidia_client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        )
    return _nvidia_client


def llm_call(prompt: str, max_retries: int = 6) -> str:
    """Send a prompt through the configured provider; never log prompts or keys."""
    if LLM_PROVIDER == "gemini":
        if not os.getenv("GEMINI_API_KEY", "").strip():
            raise RuntimeError("Gemini API key is not configured.")
        from google.genai import errors as genai_errors

        client = _get_gemini_client()
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                return (response.text or "").strip()
            except genai_errors.ClientError as exc:
                if exc.code in (429, 503) and attempt < max_retries - 1:
                    delay = 15 if exc.code == 503 else 30
                    match = re.search(r"retry in (\d+)", str(exc))
                    if match:
                        delay = int(match.group(1)) + 2
                    log.warning("[gemini] Temporary service error; retrying in %ds.", delay)
                    time.sleep(delay)
                else:
                    raise
        raise RuntimeError("Gemini call failed after max retries")

    if LLM_PROVIDER == "nvidia":
        if not os.getenv("NVIDIA_API_KEY", "").strip():
            raise RuntimeError("NVIDIA API key is not configured.")
        from openai import RateLimitError

        client = _get_nvidia_client()
        model = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=8192,
                )
                message = response.choices[0].message
                if message.content is not None:
                    return message.content.strip()
                reasoning = getattr(message, "reasoning_content", None) or ""
                if reasoning:
                    for line in reversed(reasoning.strip().splitlines()):
                        line = line.strip()
                        if line and not line.startswith(("*", "-", "#", "1.", "2.", "3.")):
                            return line
                    return reasoning.strip()
                raise RuntimeError("NVIDIA returned an empty response.")
            except RateLimitError as exc:
                if attempt < max_retries - 1:
                    delay = 30
                    match = re.search(r"retry after (\d+)", str(exc), re.IGNORECASE)
                    if match:
                        delay = int(match.group(1)) + 2
                    log.warning("[nvidia] Rate limit; retrying in %ds.", delay)
                    time.sleep(delay)
                else:
                    raise
        raise RuntimeError("NVIDIA call failed after max retries")

    raise RuntimeError(
        f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}. Set it to 'gemini' or 'nvidia'."
    )
