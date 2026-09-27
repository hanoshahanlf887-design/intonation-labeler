from __future__ import annotations

import json

from .io import FeatureData


DEMO_NOTICE = "Demo result — no Gemini API call was made."


def build_demo_result(feature_data: FeatureData) -> str:
    words = [anchor.get("word", "") for anchor in feature_data.anchors.values() if isinstance(anchor, dict)]
    first_anchor = words[0] if words else "anchor"
    return f"{DEMO_NOTICE}\nMock annotation for {first_anchor}: {feature_data.transcript}【上升】"


def build_demo_model_response(feature_data: FeatureData, low_confidence: bool = False) -> str:
    confidence = 0.45 if low_confidence else 0.82
    annotations = []
    for anchor_id, anchor in feature_data.anchors.items():
        word = anchor.get("word", anchor_id) if isinstance(anchor, dict) else anchor_id
        annotations.append(
            {
                "anchor_id": anchor_id,
                "label": "RISING",
                "confidence": confidence,
                "reason": f"{DEMO_NOTICE} Mock label for {word}.",
            }
        )
    data = {
        "transcript": feature_data.transcript,
        "annotated_text": f"{DEMO_NOTICE}\n{feature_data.transcript}【上升】",
        "annotations": annotations,
    }
    return json.dumps(data, ensure_ascii=False, indent=2)
