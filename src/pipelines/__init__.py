"""
Pipelines package for Propensity ML System.
"""
from src.pipelines.batch_pipeline import run_batch_pipeline
from src.pipelines.train_pipeline import run_training_pipeline

__all__ = ["run_batch_pipeline", "run_training_pipeline"]
