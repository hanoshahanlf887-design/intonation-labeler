from __future__ import annotations

import os
import time
from pathlib import Path


class GeminiConfigurationError(RuntimeError):
    """Raised when the Gemini client cannot be configured."""


def get_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise GeminiConfigurationError("GOOGLE_API_KEY is not set.")
    return api_key


def create_client(api_key: str | None = None):
    try:
        from google import genai
    except ImportError as exc:
        raise GeminiConfigurationError("google-genai is not installed. Run: pip install -r requirements.txt") from exc
    return genai.Client(api_key=api_key or get_api_key())


def annotate_audio(client, model_id: str, wav_path: str | Path, prompt: str) -> str:
    try:
        from google.genai import types
    except ImportError as exc:
        raise GeminiConfigurationError("google-genai is not installed. Run: pip install -r requirements.txt") from exc

    uploaded_file = None
    try:
        uploaded_file = client.files.upload(file=str(wav_path))
        while client.files.get(name=uploaded_file.name).state.name != "ACTIVE":
            time.sleep(2)

        response = client.models.generate_content(
            model=model_id,
            contents=[
                types.Part.from_uri(file_uri=uploaded_file.uri, mime_type="audio/wav"),
                types.Part.from_text(text=prompt),
            ],
        )
        return (response.text or "").strip()
    finally:
        if uploaded_file is not None:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass
