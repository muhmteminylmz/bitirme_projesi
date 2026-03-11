"""
Tests for the Twin Transition project.

These tests use synthetic data to validate the preprocessing, model training,
and visualization pipeline without requiring internet access.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make sure src is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.preprocessor import DataPreprocessor
from src.models.ml_models import ModelTrainer
from src.visualization.shap_analysis import SHAPAnalyzer


# ---------------------------------------------------------------------------
# Lightweight ModelTrainer for fast tests
# ---------------------------------------------------------------------------

class FastModelTrainer(ModelTrainer):
    """
    Overrides base learner configs with minimal settings so tests run quickly.
    """

    def _build_random_forest(self):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42, n_jobs=-1)

    def _build_xgboost(self):
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=10, max_depth=3, random_state=42, n_jobs=-1, verbosity=0)

    def _build_stacking(self):
        from sklearn.ensemble import StackingRegressor
        from sklearn.neural_network import MLPRegressor
        base_learners = [
            ("xgb", self._build_xgboost()),
            ("rf", self._build_random_forest()),
        ]
        meta_learner = MLPRegressor(
            hidden_layer_sizes=(16,), max_iter=50, random_state=42
        )
        return StackingRegressor(estimators=base_learners, final_estimator=meta_learner, cv=2)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_eurostat_df():
    """Synthetic Eurostat panel data (28 countries × 5 sectors × 10 years)."""
    rng = np.random.default_rng(0)
    countries = [
        "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES",
        "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
        "NL", "PL", "PT", "RO", "SE", "SI", "SK", "TR",
    ]
    sectors = ["C", "F", "G", "H", "J"]
    years = list(range(2014, 2024))
    # 28 × 5 × 10 = 1400 rows — covers the ≥1000 row requirement

    rows = []
    for country in countries:
        base = rng.uniform(1000, 50000)
        for sector in sectors:
            sf = {"C": 1.0, "F": 0.3, "G": 0.4, "H": 0.5, "J": 0.1}[sector]
            for year in years:
                rows.append({
                    "country": country,
                    "year": year,
                    "sector": sector,
                    "ghg_emissions": max(base * sf + rng.normal(0, base * sf * 0.1), 50),
                    "erp_usage": rng.uniform(20, 85),
                    "cloud_usage": rng.uniform(10, 70),
                    "ict_employment": rng.uniform(2, 8),
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_worldbank_df():
    """Synthetic World Bank panel data (28 countries × 10 years)."""
    rng = np.random.default_rng(1)
    countries = [
        "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES",
        "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
        "NL", "PL", "PT", "RO", "SE", "SI", "SK", "TR",
    ]
    years = list(range(2014, 2024))
    rows = []
    for country in countries:
        for year in years:
            rows.append({
                "country": country,
                "year": year,
                "gdp_per_capita": rng.uniform(10000, 80000),
                "renewable_energy_share": rng.uniform(5, 60),
                "urbanization_rate": rng.uniform(50, 95),
                "fdi_net_inflows": rng.uniform(-2, 10),
                "industry_share": rng.uniform(10, 40),
                "trade_share": rng.uniform(40, 200),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestDataPreprocessor:
    def test_merge_returns_nonempty(self, synthetic_eurostat_df, synthetic_worldbank_df, tmp_path):
        preprocessor = DataPreprocessor(output_dir=str(tmp_path))
        merged = preprocessor.merge_datasets(synthetic_eurostat_df, synthetic_worldbank_df)
        assert not merged.empty, "Merged DataFrame should not be empty."

    def test_merged_columns_present(self, synthetic_eurostat_df, synthetic_worldbank_df, tmp_path):
        preprocessor = DataPreprocessor(output_dir=str(tmp_path))
        merged = preprocessor.merge_datasets(synthetic_eurostat_df, synthetic_worldbank_df)
        expected = {
            "country", "year", "sector", "ghg_emissions",
            "erp_usage", "cloud_usage", "ict_employment",
            "gdp_per_capita", "renewable_energy_share",
        }
        assert expected.issubset(set(merged.columns))

    def test_clean_and_interpolate_no_nan_in_target(
        self, synthetic_eurostat_df, synthetic_worldbank_df, tmp_path
    ):
        preprocessor = DataPreprocessor(output_dir=str(tmp_path))
        merged = preprocessor.merge_datasets(synthetic_eurostat_df, synthetic_worldbank_df)
        cleaned = preprocessor.clean_and_interpolate(merged)
        assert cleaned["ghg_emissions"].isna().sum() == 0

    def test_encode_categoricals_adds_dummies(
        self, synthetic_eurostat_df, synthetic_worldbank_df, tmp_path
    ):
        preprocessor = DataPreprocessor(output_dir=str(tmp_path))
        merged = preprocessor.merge_datasets(synthetic_eurostat_df, synthetic_worldbank_df)
        cleaned = preprocessor.clean_and_interpolate(merged)
        encoded = preprocessor.encode_categoricals(cleaned)
        sector_dummy_cols = [c for c in encoded.columns if c.startswith("sector_")]
        assert len(sector_dummy_cols) >= 5, "Should have sector dummy columns for C,F,G,H,J"

    def test_prepare_returns_correct_shapes(
        self, synthetic_eurostat_df, synthetic_worldbank_df, tmp_path
    ):
        preprocessor = DataPreprocessor(output_dir=str(tmp_path))
        X, y, feature_names = preprocessor.prepare(synthetic_eurostat_df, synthetic_worldbank_df)
        assert len(X) >= 1000, f"Expected ≥1000 rows, got {len(X)}"
        assert len(X) == len(y), "X and y must have the same length."
        assert len(feature_names) > 0, "Feature names list must not be empty."

    def test_features_scaled_0_to_1(
        self, synthetic_eurostat_df, synthetic_worldbank_df, tmp_path
    ):
        preprocessor = DataPreprocessor(output_dir=str(tmp_path))
        X, y, feature_names = preprocessor.prepare(synthetic_eurostat_df, synthetic_worldbank_df)
        assert X.min().min() >= -1e-9, "Feature min should be ≥ 0 after MinMaxScaler."
        assert X.max().max() <= 1 + 1e-9, "Feature max should be ≤ 1 after MinMaxScaler."

    def test_target_not_scaled(
        self, synthetic_eurostat_df, synthetic_worldbank_df, tmp_path
    ):
        preprocessor = DataPreprocessor(output_dir=str(tmp_path))
        X, y, feature_names = preprocessor.prepare(synthetic_eurostat_df, synthetic_worldbank_df)
        assert y.max() > 1.0, "Target (GHG emissions) should not be scaled."

    def test_empty_eurostat_returns_empty(self, tmp_path):
        preprocessor = DataPreprocessor(output_dir=str(tmp_path))
        X, y, feature_names = preprocessor.prepare(pd.DataFrame(), pd.DataFrame())
        assert X.empty and y.empty and feature_names == []

    def test_missing_worldbank_still_works(self, synthetic_eurostat_df, tmp_path):
        preprocessor = DataPreprocessor(output_dir=str(tmp_path))
        merged = preprocessor.merge_datasets(synthetic_eurostat_df, pd.DataFrame())
        cleaned = preprocessor.clean_and_interpolate(merged)
        assert not cleaned.empty


# ---------------------------------------------------------------------------
# Model training tests
# ---------------------------------------------------------------------------

class TestModelTrainer:
    @pytest.fixture(scope="class")
    def trained_trainer_and_data(self, synthetic_eurostat_df, synthetic_worldbank_df, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("model_data")
        preprocessor = DataPreprocessor(output_dir=str(tmp_path))
        X, y, feature_names = preprocessor.prepare(synthetic_eurostat_df, synthetic_worldbank_df)
        trainer = FastModelTrainer(output_dir=str(tmp_path))
        results = trainer.train_all(X, y, feature_names)
        return trainer, results, X, y, feature_names, tmp_path

    def test_train_all_returns_four_models(self, trained_trainer_and_data):
        trainer, results, *_ = trained_trainer_and_data
        assert len(results) == 4, "Should have results for 4 models."

    def test_expected_model_names(self, trained_trainer_and_data):
        trainer, results, *_ = trained_trainer_and_data
        expected_names = {
            "Linear Regression", "Random Forest", "XGBoost", "Stacking Regressor"
        }
        assert expected_names == set(results.keys())

    def test_cv_r2_positive_for_tree_models(self, trained_trainer_and_data):
        trainer, results, *_ = trained_trainer_and_data
        for model_name in ("Random Forest", "XGBoost"):
            assert results[model_name]["cv_r2_mean"] > 0, (
                f"{model_name} CV R² should be positive on synthetic data."
            )

    def test_best_model_is_set(self, trained_trainer_and_data):
        trainer, *_ = trained_trainer_and_data
        assert trainer.best_model is not None
        assert trainer.best_model_name != ""

    def test_predict_returns_correct_shape(self, trained_trainer_and_data):
        trainer, results, X, y, feature_names, tmp_path = trained_trainer_and_data
        preds = trainer.predict(X)
        assert preds.shape == (len(X),)

    def test_stacking_regressor_has_xgb_and_rf_base_learners(self, trained_trainer_and_data):
        trainer, *_ = trained_trainer_and_data
        stacking = trainer.models["Stacking Regressor"]
        base_names = [name for name, _ in stacking.estimators]
        assert "xgb" in base_names
        assert "rf" in base_names

    def test_results_csv_saved(self, trained_trainer_and_data):
        trainer, results, X, y, feature_names, tmp_path = trained_trainer_and_data
        assert (tmp_path / "model_comparison.csv").exists()


# ---------------------------------------------------------------------------
# SHAP / Visualization tests
# ---------------------------------------------------------------------------

class TestSHAPAnalyzer:
    @pytest.fixture(scope="class")
    def trained_model_and_data(self, synthetic_eurostat_df, synthetic_worldbank_df, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("shap_data")
        preprocessor = DataPreprocessor(output_dir=str(tmp_path))
        X, y, feature_names = preprocessor.prepare(synthetic_eurostat_df, synthetic_worldbank_df)
        trainer = FastModelTrainer(output_dir=str(tmp_path))
        results = trainer.train_all(X, y, feature_names)
        return trainer, results, X, y, feature_names, tmp_path

    def test_shap_values_computed(self, trained_model_and_data):
        trainer, results, X, y, feature_names, tmp_path = trained_model_and_data
        analyzer = SHAPAnalyzer(output_dir=str(tmp_path / "figures"))
        sv = analyzer.compute_shap_values(trainer.best_model, X, "test_model")
        assert sv is not None, "SHAP values should not be None."
        assert np.array(sv).shape[1] == len(feature_names)

    def test_feature_importance_bar_plot_saved(self, trained_model_and_data):
        trainer, results, X, y, feature_names, tmp_path = trained_model_and_data
        fig_dir = tmp_path / "figures"
        analyzer = SHAPAnalyzer(output_dir=str(fig_dir))
        sv = analyzer.compute_shap_values(trainer.best_model, X, "test_model")
        analyzer.plot_feature_importance_bar(feature_names, sv)
        assert (fig_dir / "shap_feature_importance_bar.png").exists()

    def test_actual_vs_predicted_plot_saved(self, trained_model_and_data):
        trainer, results, X, y, feature_names, tmp_path = trained_model_and_data
        fig_dir = tmp_path / "figures"
        analyzer = SHAPAnalyzer(output_dir=str(fig_dir))
        y_pred = trainer.best_model.predict(X)
        analyzer.plot_actual_vs_predicted(y, y_pred, "Test Model", r2=0.95)
        assert (fig_dir / "actual_vs_predicted_test_model.png").exists()

    def test_residuals_plot_saved(self, trained_model_and_data):
        trainer, results, X, y, feature_names, tmp_path = trained_model_and_data
        fig_dir = tmp_path / "figures"
        analyzer = SHAPAnalyzer(output_dir=str(fig_dir))
        y_pred = trainer.best_model.predict(X)
        analyzer.plot_residuals(y, y_pred, "Test Model")
        assert (fig_dir / "residuals_test_model.png").exists()

    def test_model_comparison_plot_saved(self, trained_model_and_data):
        trainer, results, X, y, feature_names, tmp_path = trained_model_and_data
        fig_dir = tmp_path / "figures"
        analyzer = SHAPAnalyzer(output_dir=str(fig_dir))
        analyzer.plot_model_comparison(trainer.get_results_summary())
        assert (fig_dir / "model_comparison_r2.png").exists()
