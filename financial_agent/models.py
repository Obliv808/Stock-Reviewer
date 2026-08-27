"""Model training, evaluation and persistence.

Time-series aware: the last `split` fraction of bars is held out as an unseen
test window (no shuffling), so reported accuracy reflects realistic out-of-sample
performance. The live recommendation always uses the most recent bar.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _make_model(name: str, random_state: int):
    if name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=random_state,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=400, max_depth=6, random_state=random_state, n_jobs=-1
        )
    if name == "logistic":
        return LogisticRegression(max_iter=2000, random_state=random_state)
    raise ValueError(
        f"Unknown model {name!r}. Choose from: gradient_boosting, random_forest, logistic"
    )


@dataclass
class ModelBundle:
    ticker: str
    model_name: str
    model: object
    feature_names: List[str]
    metrics: Dict[str, float]
    train_window: Dict[str, str]          # {'start': ..., 'end': ...}
    trained_at: str = ""

    def predict_up(self, row: pd.Series) -> float:
        """P(next close > today's close) for a single feature row."""
        x = row[self.feature_names].to_frame().T
        if x.isna().any().any():
            raise ValueError("Feature row contains NaN — not enough history yet.")
        return float(self.model.predict_proba(x)[0, 1])

    def predict_up_series(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    # -- persistence ---------------------------------------------------------
    def save(self, models_dir: str) -> str:
        os.makedirs(models_dir, exist_ok=True)
        path = os.path.join(
            models_dir, f"{self.ticker.upper()}_{self.model_name}.joblib"
        )
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "metrics": self.metrics,
                "train_window": self.train_window,
                "ticker": self.ticker,
                "model_name": self.model_name,
                "trained_at": self.trained_at,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, models_dir: str, ticker: str, model_name: str) -> Optional["ModelBundle"]:
        path = os.path.join(models_dir, f"{ticker.upper()}_{model_name}.joblib")
        if not os.path.exists(path):
            return None
        d = joblib.load(path)
        return cls(**d)


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    ticker: str,
    model_name: str = "gradient_boosting",
    split: float = 0.80,
    random_state: int = 42,
) -> ModelBundle:
    n = len(X)
    cut = int(n * split)
    if cut < 100 or n - cut < 30:
        raise ValueError(f"Dataset too small for an honest train/test split (n={n}).")

    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]

    model = _make_model(model_name, random_state)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    base_acc = float(max(y_test.mean(), 1 - y_test.mean()))

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "baseline_accuracy": base_acc,
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    if len(np.unique(y_test)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba))

    return ModelBundle(
        ticker=ticker.upper(),
        model_name=model_name,
        model=model,
        feature_names=list(X.columns),
        metrics=metrics,
        train_window={
            "start": str(X_train.index[0].date()),
            "end": str(X_test.index[-1].date()),
        },
        trained_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    )


def feature_importance(bundle: ModelBundle, top_n: int = 8) -> List[tuple]:
    m = bundle.model
    if hasattr(m, "feature_importances_"):
        imp = m.feature_importances_
    elif hasattr(m, "coef_"):  # logistic
        imp = np.abs(m.coef_[0])
    else:
        return []
    order = np.argsort(imp)[::-1][:top_n]
    return [(bundle.feature_names[i], float(imp[i])) for i in order]
