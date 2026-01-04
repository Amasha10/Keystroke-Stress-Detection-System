from __future__ import annotations

import math
import pandas as pd

from schemas import KeystrokeRequest, KeystrokeEvent
from services.feature_extraction import events_to_raw_df, extract_session_features


def test_feature_extraction_smoke() -> None:
    req = KeystrokeRequest(
        user_id="u1",
        session_id=1,
        events=[
            KeystrokeEvent(key="a", press_time=0.0, release_time=0.1),
            KeystrokeEvent(key="b", press_time=0.2, release_time=0.3),
            KeystrokeEvent(key="Backspace", press_time=0.5, release_time=0.6),
        ],
    )
    raw = events_to_raw_df(req)
    feat = extract_session_features(raw)

    assert isinstance(raw, pd.DataFrame)
    assert isinstance(feat, pd.DataFrame)
    assert feat.shape[0] == 1 
    assert "typing_speed_cps" in feat.columns
    assert not math.isnan(float(feat["hold_mean"].iloc[0]))
