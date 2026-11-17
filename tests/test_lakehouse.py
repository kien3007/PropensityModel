"""
Unit tests for the Open-Source Lakehouse Layer.
Verifies Bronze ingestion, Silver views, Gold marts, and Data Quality framework.
"""
from pathlib import Path
import pytest
from src.lakehouse.engine import LakehouseEngine


@pytest.fixture
def temp_lakehouse(tmp_path):
    raw_dir = Path("sample_data/dunnhumby")
    engine = LakehouseEngine(
        lakehouse_path=tmp_path / "test_warehouse.duckdb",
        raw_data_dir=raw_dir,
    )
    return engine


def test_lakehouse_pipeline_and_dq(temp_lakehouse):
    # 1. Ingest Bronze
    ingested = temp_lakehouse.ingest_bronze()
    assert "transactions.parquet" in ingested
    assert (temp_lakehouse.bronze_dir / "transactions.parquet").exists()

    # 2. Build Silver Layer
    temp_lakehouse.build_silver_layer()

    # 3. Build Gold Layer
    temp_lakehouse.build_gold_layer()

    # 4. Check Data Quality Framework
    dq = temp_lakehouse.run_data_quality_framework()
    assert dq["status"] == "PASS"
    assert dq["details"]["dim_households_count"] == 2500
    assert dq["details"]["negative_sales_count"] == 0
