from __future__ import annotations

from typing import Any, List, Union
import pandas as pd


def _events_to_dataframe(events: Union[List[Any], pd.DataFrame]) -> pd.DataFrame:
    if isinstance(events, pd.DataFrame):
        return events.copy()

    if not isinstance(events, list):
        raise TypeError("events must be a list or a pandas DataFrame")

    if len(events) == 0:
        return pd.DataFrame(columns=["key", "press_time", "release_time", "is_backspace"])

    first = events[0]

    if hasattr(first, "model_dump"):
        rows = [e.model_dump() for e in events]
    elif hasattr(first, "dict"):
        rows = [e.dict() for e in events]
    elif isinstance(first, dict):
        rows = events
    else:
        raise TypeError("Unsupported event type")

    return pd.DataFrame(rows)


def extract_session_features(events: Union[List[Any], pd.DataFrame]) -> pd.DataFrame:
    df = _events_to_dataframe(events)

    rename_map = {
        "press_time": "Press_Time",
        "release_time": "Release_Time",
        "key": "Key",
        "is_backspace": "is_backspace",
    }
    df = df.rename(columns=rename_map)

    for col in ["Press_Time", "Release_Time"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    for col in ["Press_Time", "Release_Time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Press_Time", "Release_Time"]).reset_index(drop=True)

    if df.empty:
        return pd.DataFrame([{
            "dwell_mean": 0.0,
            "dwell_std": 0.0,
            "flight_mean": 0.0,
            "flight_std": 0.0,
            "dd_mean": 0.0,
            "dd_std": 0.0,
            "pause_mean": 0.0,
            "pause_std": 0.0,
            "typing_speed_cps": 0.0,
            "backspace_count": 0,
            "backspace_ratio": 0.0,
            "long_pause_ratio": 0.0,
        }])

    df = df.sort_values("Press_Time").reset_index(drop=True)

    df["Dwell"] = (df["Release_Time"] - df["Press_Time"]).clip(lower=0)
    df["DD"] = df["Press_Time"].diff()
    df["Flight"] = df["Press_Time"] - df["Release_Time"].shift(1)
    df["Pause"] = df["DD"]

    if "is_backspace" not in df.columns:
        df["is_backspace"] = False

    backspace_count = int(df["is_backspace"].fillna(False).astype(bool).sum())
    total_keys = int(len(df))
    backspace_ratio = float(backspace_count / total_keys) if total_keys > 0 else 0.0

    duration = float(df["Press_Time"].iloc[-1] - df["Press_Time"].iloc[0])
    typing_speed_cps = float(total_keys / duration) if duration > 0 else 0.0

    long_pause_threshold = 1.0
    long_pause_ratio = float((df["Pause"] > long_pause_threshold).sum() / total_keys) if total_keys > 0 else 0.0

    features = {
        "dwell_mean": float(df["Dwell"].mean()),
        "dwell_std": float(df["Dwell"].std(ddof=0)),
        "flight_mean": float(df["Flight"].mean()),
        "flight_std": float(df["Flight"].std(ddof=0)),
        "dd_mean": float(df["DD"].mean()),
        "dd_std": float(df["DD"].std(ddof=0)),
        "pause_mean": float(df["Pause"].mean()),
        "pause_std": float(df["Pause"].std(ddof=0)),
        "typing_speed_cps": typing_speed_cps,   
        "backspace_count": backspace_count,
        "backspace_ratio": backspace_ratio,
        "long_pause_ratio": long_pause_ratio,
    }

    return pd.DataFrame([features])


def extract_features(events: Union[List[Any], pd.DataFrame]) -> pd.DataFrame:
    return extract_session_features(events)
