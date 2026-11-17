"""
Model Training and Optimization Module.
Handles LightGBM classifier training and baseline logistic regression model.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.pit_extractor import MODEL_FEATURES

logger = logging.getLogger(__name__)

DEFAULT_LGBM_PARAMS: Dict[str, Any] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 100,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "random_state": 42,
    "n_jobs": 2,
    "verbosity": -1,
}


class ModelTrainer:
    """
    Trains propensity classification models with scikit-learn compatible interfaces.
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        lgbm_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.features = features or MODEL_FEATURES
        self.lgbm_params = lgbm_params or DEFAULT_LGBM_PARAMS

    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series | np.ndarray) -> lgb.LGBMClassifier:
        """
        Trains LightGBM classifier on extracted features.
        """
        X = X_train[self.features]
        y = np.asarray(y_train, dtype=int)

        logger.info(f"Training LightGBM on {len(X)} samples with {len(self.features)} features...")
        model = lgb.LGBMClassifier(**self.lgbm_params)
        model.fit(X, y)
        logger.info("LightGBM training completed successfully.")
        return model

    def train_baseline_logistic(self, X_train: pd.DataFrame, y_train: pd.Series | np.ndarray) -> Pipeline:
        """
        Trains baseline Logistic Regression pipeline for benchmarking.
        """
        X = X_train[self.features]
        y = np.asarray(y_train, dtype=int)

        logger.info("Training baseline Logistic Regression model...")
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        pipe.fit(X, y)
        logger.info("Baseline Logistic Regression training completed.")
        return pipe

    def get_feature_importances(self, model: lgb.LGBMClassifier) -> pd.DataFrame:
        """
        Returns feature importances sorted descending.
        """
        importances = model.feature_importances_
        df = pd.DataFrame({
            "feature": self.features,
            "importance": importances,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        return df
