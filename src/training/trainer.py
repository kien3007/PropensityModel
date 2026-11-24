"""
Model Training: LightGBM champion candidate (native categoricals, early stopping on the
validation split).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config.model_config import DEFAULT_TRAINING_CONFIG
from src.features.contract import CATEGORICAL_FEATURES, MODEL_FEATURES

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(
        self,
        features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        lgbm_params: Optional[Dict[str, Any]] = None,
        early_stopping_rounds: int = DEFAULT_TRAINING_CONFIG.early_stopping_rounds,
    ) -> None:
        self.features = list(features or MODEL_FEATURES)
        default_cats = CATEGORICAL_FEATURES if features is None else []
        cats = categorical_features if categorical_features is not None else default_cats
        self.categorical_features = [c for c in cats if c in self.features]
        self.lgbm_params = dict(lgbm_params or DEFAULT_TRAINING_CONFIG.lgbm_params)
        self.early_stopping_rounds = early_stopping_rounds

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[self.features].copy()
        for col in self.categorical_features:
            if out[col].dtype.name != "category":
                out[col] = out[col].astype("category")
        return out

    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
    ) -> lgb.LGBMClassifier:
        """Fits LightGBM; with a validation set, stops early on validation log loss."""
        X = self._prepare(X_train)
        y = np.asarray(y_train, dtype=int)
        model = lgb.LGBMClassifier(**self.lgbm_params)

        fit_kwargs: Dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_X"] = (self._prepare(X_val),)
            fit_kwargs["eval_y"] = (np.asarray(y_val, dtype=int),)
            fit_kwargs["eval_metric"] = "binary_logloss"
            fit_kwargs["callbacks"] = [lgb.early_stopping(self.early_stopping_rounds, first_metric_only=True, verbose=False)]

        logger.info(f"Training LightGBM on {len(X):,} rows x {len(self.features)} features...")
        model.fit(X, y, **fit_kwargs)
        logger.info(f"LightGBM training finished (best iteration: {self.best_iteration(model)}).")
        return model

    @staticmethod
    def best_iteration(model: lgb.LGBMClassifier) -> int:
        best = getattr(model, "best_iteration_", None)
        return int(best) if best else int(model.n_estimators)

    def predict_raw(self, model: lgb.LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
        return model.predict_proba(self._prepare(X))[:, 1]

    def get_feature_importances(self, model: lgb.LGBMClassifier, importance_type: str = "gain") -> pd.DataFrame:
        importances = model.booster_.feature_importance(importance_type=importance_type)
        return (
            pd.DataFrame({"feature": self.features, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
