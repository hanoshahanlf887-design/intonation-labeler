from __future__ import annotations

from .io import FeatureData


DEMO_NOTICE = "Demo result — no Gemini API call was made."


def build_demo_result(feature_data: FeatureData) -> str:
    words = [anchor.get("word", "") for anchor in feature_data.anchors.values() if isinstance(anchor, dict)]
    first_anchor = words[0] if words else "anchor"
    return f"{DEMO_NOTICE}\nMock annotation for {first_anchor}: {feature_data.transcript}【上升】"
