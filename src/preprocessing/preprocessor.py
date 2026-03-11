"""
Data Preprocessing Module

Handles:
1. Merging Eurostat + World Bank panel data into a single analysis-ready dataset.
2. Filtering to target countries, years, and NACE sectors.
3. Handling missing values via linear interpolation (within country-sector groups).
4. Scaling all independent variables to [0, 1] using MinMaxScaler.
5. Outputting clean feature matrix X and target vector y.
"""

import logging
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TARGET_COUNTRIES = [
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES",
    "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
    "NL", "PL", "PT", "RO", "SE", "SI", "SK", "TR",
]
TARGET_SECTORS = ["C", "F", "G", "H", "J"]
TARGET_YEARS = list(range(2014, 2024))

TARGET_VARIABLE = "ghg_emissions"

FEATURE_COLUMNS = [
    # Digitalization features
    "erp_usage",
    "cloud_usage",
    "ict_employment",
    # Macroeconomic control variables
    "gdp_per_capita",
    "renewable_energy_share",
    "urbanization_rate",
    "fdi_net_inflows",
    "industry_share",
    "trade_share",
]


class DataPreprocessor:
    """
    Merges, cleans, interpolates, and scales the panel dataset for ML training.
    """

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else None
        self.scaler = MinMaxScaler()
        self._feature_columns: List[str] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def merge_datasets(
        self,
        eurostat_df: pd.DataFrame,
        worldbank_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge Eurostat sectoral panel with World Bank country-year panel.

        Parameters
        ----------
        eurostat_df : pd.DataFrame
            Output of EurostatFetcher.fetch_all() — must have columns
            [country, year, sector, ghg_emissions, erp_usage, cloud_usage,
             ict_employment].
        worldbank_df : pd.DataFrame
            Output of WorldBankFetcher.fetch_all() — must have columns
            [country, year, gdp_per_capita, ...].

        Returns
        -------
        pd.DataFrame
            Merged panel with all variables.
        """
        if eurostat_df.empty:
            logger.error("Eurostat DataFrame is empty — cannot merge.")
            return pd.DataFrame()

        df = eurostat_df.copy()

        # Filter to target scope
        df = df[df["country"].isin(TARGET_COUNTRIES)]
        df = df[df["sector"].isin(TARGET_SECTORS)]
        df = df[df["year"].isin(TARGET_YEARS)]

        if not worldbank_df.empty:
            df = df.merge(worldbank_df, on=["country", "year"], how="left")
        else:
            logger.warning("World Bank DataFrame is empty — macroeconomic variables will be NaN.")
            for col in ["gdp_per_capita", "renewable_energy_share", "urbanization_rate",
                        "fdi_net_inflows", "industry_share", "trade_share"]:
                df[col] = np.nan

        logger.info("Merged dataset: %d rows, %d columns.", len(df), df.shape[1])
        return df

    def clean_and_interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using linear interpolation within each
        country-sector group (sorted by year).  Remaining NaNs at the
        boundaries are filled via forward-fill then back-fill.

        Parameters
        ----------
        df : pd.DataFrame
            Raw merged panel data.

        Returns
        -------
        pd.DataFrame
            Panel data with missing values handled.
        """
        if df.empty:
            return df

        df = df.copy()
        df = df.sort_values(["country", "sector", "year"]).reset_index(drop=True)

        numeric_cols = [c for c in FEATURE_COLUMNS + [TARGET_VARIABLE] if c in df.columns]

        logger.info("Missing values before interpolation:")
        for col in numeric_cols:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                logger.info("  %-30s %d / %d (%.1f%%)",
                            col, n_missing, len(df), 100 * n_missing / len(df))

        # Use transform so that all original columns (including group keys) are preserved.
        for col in numeric_cols:
            if col in df.columns:
                df[col] = (
                    df.groupby(["country", "sector"])[col]
                    .transform(
                        lambda s: s.interpolate(method="linear", limit_direction="both")
                        .ffill()
                        .bfill()
                    )
                )
        df = df.reset_index(drop=True)

        # Final check — drop rows where target is still NaN
        before = len(df)
        df = df.dropna(subset=[TARGET_VARIABLE])
        after = len(df)
        if before != after:
            logger.warning(
                "Dropped %d rows where %s is still NaN after interpolation.",
                before - after, TARGET_VARIABLE,
            )

        logger.info("Dataset after interpolation: %d rows.", len(df))
        return df

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode the 'sector' column so that tree-based models can use it
        as a categorical feature. The 'country' column is kept for reference but
        not encoded by default (too many unique values for tree models).

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned panel data.

        Returns
        -------
        pd.DataFrame
            Panel with sector dummy variables added.
        """
        if df.empty:
            return df

        df = df.copy()
        sector_dummies = pd.get_dummies(df["sector"], prefix="sector", drop_first=False)
        df = pd.concat([df, sector_dummies], axis=1)
        logger.info("Encoded 'sector' column into %d dummies.", sector_dummies.shape[1])
        return df

    def scale_features(
        self, df: pd.DataFrame, fit: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Scale all independent variables (features) to [0, 1] using MinMaxScaler.
        The target variable (ghg_emissions) is NOT scaled.

        Parameters
        ----------
        df : pd.DataFrame
            Panel data (should include sector dummy columns).
        fit : bool
            If True, fit the scaler on this data.
            If False, use the previously fitted scaler (for test/predict).

        Returns
        -------
        Tuple[pd.DataFrame, List[str]]
            - DataFrame with scaled features.
            - List of feature column names used.
        """
        if df.empty:
            return df, []

        df = df.copy()

        base_features = [c for c in FEATURE_COLUMNS if c in df.columns]
        sector_dummies = [c for c in df.columns if c.startswith("sector_")]
        all_features = base_features + sector_dummies

        self._feature_columns = all_features

        if fit:
            df[all_features] = self.scaler.fit_transform(df[all_features].astype(float))
        else:
            df[all_features] = self.scaler.transform(df[all_features].astype(float))

        logger.info("Scaled %d feature columns with MinMaxScaler.", len(all_features))
        return df, all_features

    def prepare(
        self,
        eurostat_df: pd.DataFrame,
        worldbank_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Full preprocessing pipeline: merge → clean → encode → scale.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series, List[str]]
            - X : feature matrix (scaled)
            - y : target vector (ghg_emissions, unscaled)
            - feature_names : list of feature column names
        """
        df = self.merge_datasets(eurostat_df, worldbank_df)
        df = self.clean_and_interpolate(df)
        df = self.encode_categoricals(df)
        df, feature_names = self.scale_features(df, fit=True)

        if df.empty or TARGET_VARIABLE not in df.columns:
            logger.error("Preprocessing failed — empty or missing target column.")
            return pd.DataFrame(), pd.Series(dtype=float), []

        X = df[feature_names].copy()
        y = df[TARGET_VARIABLE].copy()

        # Save processed dataset if output_dir is set
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.output_dir / "processed_panel_data.csv"
            processed = df.copy()
            processed[feature_names] = X
            processed.to_csv(out_path, index=False)
            logger.info("Saved processed dataset to %s", out_path)

        logger.info(
            "Preprocessing complete. X shape: %s, y shape: %s.",
            X.shape, y.shape,
        )
        logger.info("Target variable statistics:\n%s", y.describe().to_string())
        logger.info("Dataset size: %d rows — target: ≥1000 rows.", len(df))

        return X, y, feature_names
