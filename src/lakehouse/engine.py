"""
Open-Source Lakehouse Engine implemented with DuckDB and Parquet.
Follows Medallion Architecture (Bronze -> Silver -> Gold).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import duckdb
import pandas as pd

from src.config.settings import settings

logger = logging.getLogger(__name__)


class LakehouseEngine:
    """
    Manages the Open-Source Lakehouse storage and transformation engine.
    """

    def __init__(
        self,
        lakehouse_path: Optional[Path | str] = None,
        raw_data_dir: Optional[Path | str] = None,
    ) -> None:
        self.db_path = Path(lakehouse_path or settings.LAKEHOUSE_PATH)
        self.raw_data_dir = Path(raw_data_dir or settings.DATA_RAW_DIR)

        self.lakehouse_dir = self.db_path.parent
        self.bronze_dir = self.lakehouse_dir / "bronze"
        self.silver_dir = self.lakehouse_dir / "silver"
        self.gold_dir = self.lakehouse_dir / "gold"

        for d in (self.bronze_dir, self.silver_dir, self.gold_dir):
            d.mkdir(parents=True, exist_ok=True)

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path))

    def ingest_bronze(self, force_refresh: bool = False) -> Dict[str, str]:
        """
        Ingests raw CSV files into Bronze immutable Parquet files.
        """
        con = self.get_connection()
        ingested = {}

        tables = [
            ("transaction_data.csv", "transactions.parquet"),
            ("product.csv", "products.parquet"),
            ("hh_demographic.csv", "demographics.parquet"),
            ("campaign_table.csv", "campaign_table.parquet"),
            ("campaign_desc.csv", "campaign_desc.parquet"),
            ("coupon.csv", "coupons.parquet"),
            ("coupon_redempt.csv", "coupon_redemptions.parquet"),
        ]

        try:
            for csv_name, parquet_name in tables:
                csv_path = self.raw_data_dir / csv_name
                parquet_path = self.bronze_dir / parquet_name

                if not csv_path.exists():
                    logger.warning(f"File {csv_path} does not exist, skipping.")
                    continue

                if not parquet_path.exists() or force_refresh:
                    csv_escaped = str(csv_path).replace("\\", "/")
                    parquet_escaped = str(parquet_path).replace("\\", "/")

                    con.execute(f"""
                        COPY (SELECT * FROM read_csv_auto('{csv_escaped}'))
                        TO '{parquet_escaped}' (FORMAT PARQUET, COMPRESSION ZSTD);
                    """)
                    logger.info(f"Ingested {csv_name} -> {parquet_path}")

                ingested[parquet_name] = str(parquet_path)
        finally:
            con.close()

        return ingested

    def build_silver_layer(self) -> None:
        """
        Transforms Bronze Parquet into Silver cleaned views and tables.
        """
        con = self.get_connection()
        try:
            tx_parquet = str(self.bronze_dir / "transactions.parquet").replace("\\", "/")
            prod_parquet = str(self.bronze_dir / "products.parquet").replace("\\", "/")

            if not (self.bronze_dir / "transactions.parquet").exists():
                raise FileNotFoundError("Bronze transactions.parquet not found. Run ingest_bronze() first.")

            # Staging: stg_transactions
            con.execute(f"""
                CREATE OR REPLACE VIEW stg_transactions AS
                SELECT
                    household_key::BIGINT AS household_key,
                    BASKET_ID::BIGINT AS basket_id,
                    DAY::INTEGER AS day,
                    PRODUCT_ID::BIGINT AS product_id,
                    QUANTITY::INTEGER AS quantity,
                    SALES_VALUE::DOUBLE AS sales_value,
                    STORE_ID::BIGINT AS store_id,
                    RETAIL_DISC::DOUBLE AS retail_disc,
                    COUPON_DISC::DOUBLE AS coupon_disc,
                    WEEK_NO::INTEGER AS week_no
                FROM read_parquet('{tx_parquet}')
                WHERE SALES_VALUE >= 0;
            """)

            # Staging: stg_products
            if (self.bronze_dir / "products.parquet").exists():
                con.execute(f"""
                    CREATE OR REPLACE VIEW stg_products AS
                    SELECT
                        PRODUCT_ID::BIGINT AS product_id,
                        DEPARTMENT::VARCHAR AS department,
                        COMMODITY_DESC::VARCHAR AS commodity_desc,
                        BRAND::VARCHAR AS brand,
                        CASE 
                            WHEN DEPARTMENT IN ('GROCERY', 'PASTRY', 'MEAT', 'MEAT-PCKGD', 'SEAFOOD', 'PRODUCE', 'DELI', 'DAIRY')
                            THEN 1 ELSE 0 
                        END AS is_perishable
                    FROM read_parquet('{prod_parquet}');
                """)

            # Intermediate: int_household_daily
            con.execute("""
                CREATE OR REPLACE VIEW int_household_daily AS
                SELECT
                    household_key,
                    day,
                    COUNT(DISTINCT basket_id) AS baskets_count,
                    SUM(sales_value) AS daily_sales,
                    COUNT(product_id) AS line_items_count
                FROM stg_transactions
                GROUP BY household_key, day;
            """)

            logger.info("Silver views created successfully.")
        finally:
            con.close()

    def build_gold_layer(self) -> None:
        """
        Builds Gold dimensional marts and persistent scoring tables.
        """
        con = self.get_connection()
        try:
            demo_parquet = str(self.bronze_dir / "demographics.parquet").replace("\\", "/")
            # Without demographics, join an empty typed relation so the column references stay valid.
            demo_source = (
                f"read_parquet('{demo_parquet}')"
                if (self.bronze_dir / "demographics.parquet").exists()
                else "(SELECT NULL::BIGINT AS household_key, NULL::VARCHAR AS classification_1, "
                     "NULL::VARCHAR AS classification_2, NULL::VARCHAR AS classification_3, "
                     "NULL::VARCHAR AS HOMEOWNER_DESC, NULL::VARCHAR AS classification_4, "
                     "NULL::VARCHAR AS KID_CATEGORY_DESC WHERE FALSE)"
            )
            demo_sql = f"LEFT JOIN {demo_source} d ON h.household_key = d.household_key"

            con.execute(f"""
                CREATE OR REPLACE TABLE dim_households AS
                WITH all_hh AS (
                    SELECT UNNEST(range(1, 2501)) AS household_key
                )
                SELECT 
                    h.household_key,
                    COALESCE(d.classification_1, 'UNKNOWN') AS age_desc,
                    COALESCE(d.classification_2, 'U') AS marital_status,
                    COALESCE(d.classification_3, 'UNKNOWN') AS income_desc,
                    COALESCE(d.HOMEOWNER_DESC, 'Unknown') AS homeowner_desc,
                    COALESCE(d.classification_4, 'UNKNOWN') AS household_size,
                    COALESCE(d.KID_CATEGORY_DESC, 'None/Unknown') AS kid_category,
                    CASE WHEN d.household_key IS NOT NULL THEN 1 ELSE 0 END AS has_demographics
                FROM all_hh h
                {demo_sql};
            """)

            con.execute("""
                CREATE TABLE IF NOT EXISTS fct_weekly_batch_scores (
                    run_id VARCHAR,
                    snapshot_day INTEGER,
                    household_key BIGINT,
                    purchase_probability DOUBLE,
                    rank INTEGER,
                    decile INTEGER,
                    urgency_tier VARCHAR,
                    marketing_action VARCHAR,
                    scored_at TIMESTAMP,
                    PRIMARY KEY (run_id, household_key)
                );
            """)

            logger.info("Gold tables created successfully.")
        finally:
            con.close()

    def run_data_quality_framework(self) -> Dict[str, Any]:
        """
        Executes automated Data Quality checks.
        """
        con = self.get_connection()
        checks: Dict[str, Any] = {"status": "PASS", "details": {}}
        try:
            hh_count = con.execute("SELECT COUNT(*) FROM dim_households;").fetchone()[0]
            checks["details"]["dim_households_count"] = hh_count
            if hh_count != 2500:
                checks["status"] = "FAIL"
                checks["details"]["error_hh_count"] = f"Expected 2500, got {hh_count}"

            neg_sales = con.execute("SELECT COUNT(*) FROM stg_transactions WHERE sales_value < 0;").fetchone()[0]
            checks["details"]["negative_sales_count"] = neg_sales
            if neg_sales > 0:
                checks["status"] = "FAIL"
                checks["details"]["error_negative_sales"] = f"Found {neg_sales} negative sales rows"

        except Exception as e:
            checks["status"] = "ERROR"
            checks["details"]["exception"] = str(e)
        finally:
            con.close()

        return checks

    def save_batch_scores(self, scores_df: pd.DataFrame, run_id: str, snapshot_day: int) -> int:
        con = self.get_connection()
        try:
            con.execute("""
                CREATE TABLE IF NOT EXISTS fct_weekly_batch_scores (
                    run_id VARCHAR,
                    snapshot_day INTEGER,
                    household_key BIGINT,
                    purchase_probability DOUBLE,
                    rank INTEGER,
                    decile INTEGER,
                    urgency_tier VARCHAR,
                    marketing_action VARCHAR,
                    scored_at TIMESTAMP,
                    PRIMARY KEY (run_id, household_key)
                );
            """)

            df_to_save = scores_df.copy()
            df_to_save["run_id"] = str(run_id)
            df_to_save["snapshot_day"] = int(snapshot_day)

            con.register("new_scores", df_to_save)
            con.execute(f"""
                DELETE FROM fct_weekly_batch_scores WHERE run_id = '{run_id}';
                INSERT INTO fct_weekly_batch_scores
                SELECT
                    run_id,
                    snapshot_day,
                    household_key,
                    purchase_probability,
                    rank,
                    decile,
                    urgency_tier,
                    marketing_action,
                    CURRENT_TIMESTAMP AS scored_at
                FROM new_scores;
            """)
            return con.execute(f"SELECT COUNT(*) FROM fct_weekly_batch_scores WHERE run_id = '{run_id}';").fetchone()[0]
        finally:
            con.close()
