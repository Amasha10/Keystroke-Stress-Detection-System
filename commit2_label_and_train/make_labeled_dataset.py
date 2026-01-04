from __future__ import annotations
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import settings

APP_DIR = Path(__file__).resolve().parent
RAW_GLOB = str(APP_DIR / "data" / "*_keystroke_raw.csv")
OUT_LABELED = APP_DIR  / "labeled_sessions.csv"

PAUSE_THRESHOLD_S = float(settings.PAUSE_THRESHOLD_S) 
LABEL_TOP_QUANTILE = 0.70  


def load_raw() -> pd.DataFrame:
    paths = sorted(glob.glob(RAW_GLOB))
    if not paths:
        raise FileNotFoundError(
            f"No files matched {RAW_GLOB}. Put raw csv files into app/data/"
        )
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p, engine="python", on_bad_lines="skip")
        except Exception as e:
            print(f"Failed to read {p}: {e}")
            continue
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract session-level features (same feature family as the API)."""
    df = df.copy()

    # Ensure numeric
    for c in ["Press_Time", "Release_Time", "Hold_Time", "DD", "UD", "Characters_Count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["is_backspace"] = (
        df["Key_Pressed"].astype(str).str.contains("backspace", case=False, na=False).astype(int)
    )
    df["is_shift"] = (
        df["Key_Pressed"].astype(str).str.contains("shift", case=False, na=False).astype(int)
    )

    df["long_pause_dd"] = (df["DD"] > PAUSE_THRESHOLD_S).astype(int)
    df["long_pause_ud"] = (df["UD"] > PAUSE_THRESHOLD_S).astype(int)

    gcols = ["User_ID", "Session_ID"]
    g = df.groupby(gcols, dropna=False)

    def safe_mean(x):
        return float(np.nanmean(x)) if len(x) else np.nan

    def safe_std(x):
        return float(np.nanstd(x)) if len(x) else np.nan

    def safe_median(x):
        return float(np.nanmedian(x)) if len(x) else np.nan

    feat = pd.DataFrame(
        {
            "hold_mean": g["Hold_Time"].apply(safe_mean),
            "hold_std": g["Hold_Time"].apply(safe_std),
            "hold_median": g["Hold_Time"].apply(safe_median),
            "dd_mean": g["DD"].apply(safe_mean),
            "dd_std": g["DD"].apply(safe_std),
            "dd_median": g["DD"].apply(safe_median),
            "ud_mean": g["UD"].apply(safe_mean),
            "ud_std": g["UD"].apply(safe_std),
            "ud_median": g["UD"].apply(safe_median),
            "backspace_count": g["is_backspace"].sum(),
            "shift_count": g["is_shift"].sum(),
            "long_pause_dd_ratio": g["long_pause_dd"].mean(),
            "long_pause_ud_ratio": g["long_pause_ud"].mean(),
        }
    ).reset_index()

    press_min = g["Press_Time"].min().reset_index(name="press_min")
    press_max = g["Press_Time"].max().reset_index(name="press_max")
    char_max = g["Characters_Count"].max().reset_index(name="chars_total")

    feat = (
        feat.merge(press_min, on=gcols, how="left")
        .merge(press_max, on=gcols, how="left")
        .merge(char_max, on=gcols, how="left")
    )

    feat["active_time_s"] = (feat["press_max"] - feat["press_min"]).replace(0, np.nan)
    feat["typing_speed_cps"] = feat["chars_total"] / feat["active_time_s"]
    feat["backspace_ratio"] = feat["backspace_count"] / feat["chars_total"].replace(0, np.nan)

    feat = feat.drop(columns=["press_min", "press_max", "chars_total", "active_time_s"], errors="ignore")
    return feat


def make_labels(feat: pd.DataFrame) -> pd.DataFrame:
    feat = feat.copy()

    def z(x: pd.Series) -> pd.Series:
        mu = x.mean(skipna=True)
        sd = x.std(skipna=True)
        if sd == 0 or np.isnan(sd):
            return x * 0.0
        return (x - mu) / sd

    feat["stress_score"] = (
        0.35 * z(feat["long_pause_dd_ratio"])
        + 0.20 * z(feat["long_pause_ud_ratio"])
        + 0.15 * z(feat["hold_mean"])
        + 0.15 * z(feat["dd_std"])
        + 0.10 * z(feat["ud_std"])
        + 0.10 * z(feat["backspace_ratio"])
        - 0.10 * z(feat["typing_speed_cps"])
    )

    thr = feat["stress_score"].quantile(LABEL_TOP_QUANTILE)
    feat["stress_label"] = (feat["stress_score"] > thr).astype(int)
    return feat


def main() -> None:
    raw = load_raw()
    feat = extract_features(raw)
    labeled = make_labels(feat)

    OUT_LABELED.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(OUT_LABELED, index=False)

    print(f"Saved: {OUT_LABELED}")
    print("Label distribution:")
    print(labeled["stress_label"].value_counts(dropna=False))
    print("\nPreview:")
    print(labeled[["User_ID", "Session_ID", "stress_score", "stress_label"]].head(10))


if __name__ == "__main__":
    main()
