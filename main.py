"""
Twin Transition: Sectoral Digitalization and Carbon Intensity in the EU

Main execution pipeline for the graduation thesis project:
"Modelling the Effect of Sectoral Digitalization on Carbon Intensity in the
European Union's Twin Transition Using Hybrid Machine Learning and
Explainable AI (SHAP)"

Pipeline steps:
1. Fetch data from Eurostat (GHG emissions + digitalization indicators)
2. Fetch macroeconomic control variables from World Bank API
3. Merge, clean, interpolate, and scale the panel dataset
4. Train and cross-validate: Linear Regression, Random Forest, XGBoost,
   and the final Stacking Regressor (XGB + RF → MLPRegressor)
5. Generate SHAP feature importance and summary plots
6. Save all results and figures to outputs/

Usage
-----
    python main.py

    # Skip live data fetching and use cached data (if available):
    python main.py --use-cache

    # Run with synthetic demo data (no internet required):
    python main.py --demo
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Project root → ensure imports work when run from project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.eurostat_fetcher import EurostatFetcher
from src.data_collection.worldbank_fetcher import WorldBankFetcher
from src.models.ml_models import ModelTrainer
from src.preprocessing.preprocessor import DataPreprocessor
from src.visualization.shap_analysis import SHAPAnalyzer

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"
CACHE_DIR = OUTPUT_DIR / "cache"


# ---------------------------------------------------------------------------
# Synthetic demo data generator (used with --demo flag)
# ---------------------------------------------------------------------------

def _generate_demo_data():
    """
    Generate a realistic synthetic panel dataset (28 countries × 5 sectors
    × 10 years = 1400 rows) that mimics the real structure.
    Used for offline demonstration / CI testing without internet access.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)

    countries = [
        "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES",
        "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
        "NL", "PL", "PT", "RO", "SE", "SI", "SK", "TR",
    ]
    sectors = ["C", "F", "G", "H", "J"]
    years = list(range(2014, 2024))

    rows = []
    for country in countries:
        # Country-level baseline offset
        base_ghg = rng.uniform(5_000, 80_000)
        for sector in sectors:
            sector_factor = {"C": 1.0, "F": 0.3, "G": 0.4, "H": 0.5, "J": 0.15}[sector]
            for year in years:
                erp = rng.uniform(20, 85)
                cloud = rng.uniform(10, 70)
                ict = rng.uniform(2, 8)
                gdp = rng.uniform(10_000, 80_000)
                ren = rng.uniform(5, 60)
                urb = rng.uniform(50, 95)
                fdi = rng.uniform(-2, 10)
                ind = rng.uniform(10, 40)
                trade = rng.uniform(40, 200)

                # Non-linear target with interaction effects
                ghg = (
                    base_ghg * sector_factor
                    * (1 - 0.003 * erp)
                    * (1 - 0.002 * cloud)
                    * (1 - 0.04 * ren)
                    * (0.8 + 0.005 * (year - 2014))
                    + rng.normal(0, base_ghg * sector_factor * 0.05)
                )
                rows.append({
                    "country": country,
                    "year": year,
                    "sector": sector,
                    "ghg_emissions": max(ghg, 100),
                    "erp_usage": erp,
                    "cloud_usage": cloud,
                    "ict_employment": ict,
                    "gdp_per_capita": gdp,
                    "renewable_energy_share": ren,
                    "urbanization_rate": urb,
                    "fdi_net_inflows": fdi,
                    "industry_share": ind,
                    "trade_share": trade,
                })

    df = pd.DataFrame(rows)
    logger.info("Demo dataset generated: %d rows × %d columns.", *df.shape)
    return df

def run_pipeline(use_cache: bool = False, demo: bool = False) -> None:
    """
    Execute the full Twin Transition analysis pipeline.

    Parameters
    ----------
    use_cache : bool
        Load previously cached CSV files instead of re-fetching from APIs.
    demo : bool
        Use synthetic demo data (no internet required).
    """
    logger.info("=" * 70)
    logger.info("TWIN TRANSITION: Digitalization ↔ Carbon Emissions Analysis")
    logger.info("Scope: 28 countries | 5 NACE sectors | 2014–2023")
    logger.info("=" * 70)

    # Create output directories
    for d in [OUTPUT_DIR, FIGURES_DIR, RESULTS_DIR, CACHE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1: Data Collection
    # ------------------------------------------------------------------
    logger.info("\n[STEP 1] Data Collection")

    if demo:
        logger.info("Using synthetic demo data (--demo flag active).")
        full_df = _generate_demo_data()
        eurostat_df = full_df[[
            "country", "year", "sector",
            "ghg_emissions", "erp_usage", "cloud_usage", "ict_employment",
        ]].copy()
        worldbank_df = full_df[[
            "country", "year",
            "gdp_per_capita", "renewable_energy_share", "urbanization_rate",
            "fdi_net_inflows", "industry_share", "trade_share",
        ]].drop_duplicates(subset=["country", "year"]).copy()

    else:
        cache_euro = CACHE_DIR / "eurostat_raw.csv"
        cache_wb = CACHE_DIR / "worldbank_raw.csv"

        if use_cache and cache_euro.exists() and cache_wb.exists():
            logger.info("Loading cached Eurostat data from %s", cache_euro)
            import pandas as pd  # noqa: PLC0415 (local import for cache path only)
            eurostat_df = pd.read_csv(cache_euro)
            logger.info("Loading cached World Bank data from %s", cache_wb)
            worldbank_df = pd.read_csv(cache_wb)
        else:
            # Live fetch from Eurostat
            euro_fetcher = EurostatFetcher()
            eurostat_df = euro_fetcher.fetch_all()
            if not eurostat_df.empty:
                eurostat_df.to_csv(cache_euro, index=False)
                logger.info("Cached Eurostat data → %s", cache_euro)

            # Live fetch from World Bank
            wb_fetcher = WorldBankFetcher()
            worldbank_df = wb_fetcher.fetch_all()
            if not worldbank_df.empty:
                worldbank_df.to_csv(cache_wb, index=False)
                logger.info("Cached World Bank data → %s", cache_wb)

    # ------------------------------------------------------------------
    # STEP 2: Preprocessing
    # ------------------------------------------------------------------
    logger.info("\n[STEP 2] Data Preprocessing")

    preprocessor = DataPreprocessor(output_dir=str(RESULTS_DIR))
    X, y, feature_names = preprocessor.prepare(eurostat_df, worldbank_df)

    if X.empty or len(X) < 10:
        logger.error(
            "Dataset is too small (%d rows) for meaningful ML training. "
            "Check data sources or run with --demo.",
            len(X),
        )
        sys.exit(1)

    logger.info("Final dataset: %d samples, %d features.", *X.shape)
    logger.info("Feature columns: %s", feature_names)

    # ------------------------------------------------------------------
    # STEP 3: Model Training
    # ------------------------------------------------------------------
    logger.info("\n[STEP 3] Model Training & Cross-Validation")

    trainer = ModelTrainer(output_dir=str(RESULTS_DIR))
    results = trainer.train_all(X, y, feature_names)

    results_df = trainer.get_results_summary()
    results_csv = RESULTS_DIR / "model_comparison.csv"
    logger.info("Results saved to %s", results_csv)

    # ------------------------------------------------------------------
    # STEP 4: SHAP Analysis & Visualizations
    # ------------------------------------------------------------------
    logger.info("\n[STEP 4] SHAP Analysis & Visualization")

    analyzer = SHAPAnalyzer(output_dir=str(FIGURES_DIR))
    analyzer.run_full_analysis(
        model=trainer.best_model,
        X=X,
        y=y,
        feature_names=feature_names,
        model_name=trainer.best_model_name,
        results_df=results_df,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("Best Model  : %s", trainer.best_model_name)
    best_r2 = results.get(trainer.best_model_name, {}).get("cv_r2_mean", "N/A")
    logger.info("CV R²       : %s", best_r2)
    logger.info("Output dir  : %s", OUTPUT_DIR)
    logger.info("Figures dir : %s", FIGURES_DIR)
    logger.info("Results dir : %s", RESULTS_DIR)
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Twin Transition: ML & SHAP analysis of digitalization → GHG emissions."
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Load cached CSV data from outputs/cache/ instead of re-fetching from APIs.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with synthetic demo data (no internet connection required).",
    )
    args = parser.parse_args()
    run_pipeline(use_cache=args.use_cache, demo=args.demo)


if __name__ == "__main__":
    main()
