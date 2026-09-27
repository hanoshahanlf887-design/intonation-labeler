from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any


DEFAULT_RELAY_MODEL = "gemini-3.8-flash"


class RelayConfigurationError(RuntimeError):
    """Raised when the Gemini relay client cannot be configured."""


class RelayResponseError(RuntimeError):
    """Raised when the Gemini relay returns an unusable response."""


def get_relay_api_key() -> str:
    api_key = os.getenv("GEMINI_RELAY_API_KEY", "").strip()
    if not api_key:
        raise RelayConfigurationError("GEMINI_RELAY_API_KEY is not set.")
    return api_key


def get_relay_base_url() -> str:
    base_url = os.getenv("GEMINI_RELAY_BASE_URL", "").strip()
    if not base_url:
        raise RelayConfigurationError("GEMINI_RELAY_BASE_URL is not set.")
    return base_url.rstrip("/")


def annotate_audio_via_relay(
    model_id: str,
    wav_path: str | Path,
    prompt: str,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 120.0,
) -> str:
    try:
        import httpx
    except ImportError as exc:
        raise RelayConfigurationError("httpx is not installed. Run: pip install -r requirements.txt") from exc

    key = api_key or get_relay_api_key()
    endpoint = f"{(base_url or get_relay_base_url()).rstrip('/')}/v1beta/models/{model_id}:generateContent"
    audio_b64 = base64.b64encode(Path(wav_path).read_bytes()).decode("ascii")
    payload: dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inlineData": {"mimeType": "audio/wav", "data": audio_b64}},
                    {"text": prompt},
                ],
            }
        ]
    }

    response = httpx.post(
        endpoint,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RelayResponseError(_format_relay_error(response))

    data = response.json()
    texts = _extract_candidate_texts(data)
    if not texts:
        raise RelayResponseError("Relay response did not contain candidates[].content.parts[].text.")
    return "\n".join(texts).strip()


def _extract_candidate_texts(data: Any) -> list[str]:
    texts: list[str] = []
    candidates = data.get("candidates") if isinstance(data, dict) else None
    if not isinstance(candidates, list):
        return texts
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            continue
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
    return texts


def _format_relay_error(response) -> str:
    try:
        data = response.json()
    except Exception:
        return f"Relay HTTP {response.status_code}: {response.text[:500]}"
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        error = data["error"]
        message = error.get("message", "")
        kind = error.get("type") or error.get("status") or error.get("code")
        return f"Relay HTTP {response.status_code}: {kind}: {message}"
    return f"Relay HTTP {response.status_code}: {str(data)[:500]}"
