from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

APP_DIR = Path(__file__).resolve().parent
LABELED = APP_DIR / "labeled_sessions.csv"
MODEL_OUT = APP_DIR / "keystroke_stress_model.joblib"


def main() -> None:
    if not LABELED.exists():
        raise FileNotFoundError(
            f"Missing {LABELED}. If you only have raw data, run make_labeled_dataset.py first."
        )

    df = pd.read_csv(LABELED)

    target_col = "stress_label"
    ignore_cols = {"User_ID", "Session_ID", target_col, "stress_score"}
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    X = df[feature_cols]
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(classification_report(y_test, preds, digits=4))

    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_OUT)
    print(f"Saved model: {MODEL_OUT}")


if __name__ == "__main__":
    main()
