"""
Machine Learning Models Module

Trains and evaluates the following models on the Twin Transition panel dataset:
1. Linear Regression (baseline / traditional econometric)
2. Random Forest Regressor
3. XGBoost Regressor
4. Stacking Regressor
   - Base learners : XGBoost + Random Forest
   - Meta-learner  : MLPRegressor (Neural Network)

Uses 5-fold cross-validation and reports R², RMSE, and MAE for each model.
The best model (Stacking Regressor) is saved for SHAP analysis.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
CV_FOLDS = 5


class ModelTrainer:
    """
    Builds, cross-validates, and evaluates all ML models described in the
    Twin Transition project methodology.
    """

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else None
        self.models: Dict[str, object] = {}
        self.cv_results: Dict[str, Dict[str, float]] = {}
        self.best_model = None
        self.best_model_name: str = ""

    # ------------------------------------------------------------------
    # Model definitions
    # ------------------------------------------------------------------

    def _build_linear_regression(self) -> LinearRegression:
        return LinearRegression()

    def _build_random_forest(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    def _build_xgboost(self) -> XGBRegressor:
        return XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )

    def _build_stacking(self) -> StackingRegressor:
        """
        Stacking Regressor:
        - Base learners: XGBoost + Random Forest
        - Meta-learner : MLPRegressor (Artificial Neural Network)
        """
        base_learners = [
            ("xgb", self._build_xgboost()),
            ("rf", self._build_random_forest()),
        ]
        meta_learner = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        )
        return StackingRegressor(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=CV_FOLDS,
            n_jobs=-1,
        )

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate(
        model,
        X: pd.DataFrame,
        y: pd.Series,
        prefix: str = "test",
    ) -> Dict[str, float]:
        """Compute R², RMSE, and MAE for a fitted model."""
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        return {
            f"{prefix}_r2": round(r2, 4),
            f"{prefix}_rmse": round(rmse, 4),
            f"{prefix}_mae": round(mae, 4),
        }

    def _cross_validate(
        self, model, X: pd.DataFrame, y: pd.Series, model_name: str
    ) -> Dict[str, float]:
        """Run K-fold cross-validation and return averaged metrics."""
        kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_r2 = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=-1)
        cv_neg_rmse = cross_val_score(
            model, X, y, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        cv_neg_mae = cross_val_score(
            model, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        results = {
            "cv_r2_mean": round(cv_r2.mean(), 4),
            "cv_r2_std": round(cv_r2.std(), 4),
            "cv_rmse_mean": round((-cv_neg_rmse).mean(), 4),
            "cv_rmse_std": round((-cv_neg_rmse).std(), 4),
            "cv_mae_mean": round((-cv_neg_mae).mean(), 4),
            "cv_mae_std": round((-cv_neg_mae).std(), 4),
        }
        logger.info(
            "[%s] CV R²: %.4f ± %.4f  |  RMSE: %.4f ± %.4f  |  MAE: %.4f ± %.4f",
            model_name,
            results["cv_r2_mean"], results["cv_r2_std"],
            results["cv_rmse_mean"], results["cv_rmse_std"],
            results["cv_mae_mean"], results["cv_mae_std"],
        )
        return results

    # ------------------------------------------------------------------
    # Training pipeline
    # ------------------------------------------------------------------

    def train_all(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all four models and collect cross-validation results.

        Parameters
        ----------
        X : pd.DataFrame
            Scaled feature matrix.
        y : pd.Series
            Target variable (GHG emissions).
        feature_names : list of str, optional
            Names for features (used for logging).

        Returns
        -------
        dict
            Dictionary mapping model name → metric dict.
        """
        if X.empty or y.empty:
            logger.error("X or y is empty — cannot train models.")
            return {}

        logger.info("=" * 60)
        logger.info("Starting model training on %d samples, %d features.", *X.shape)
        logger.info("=" * 60)

        model_builders = {
            "Linear Regression": self._build_linear_regression,
            "Random Forest": self._build_random_forest,
            "XGBoost": self._build_xgboost,
            "Stacking Regressor": self._build_stacking,
        }

        all_results: Dict[str, Dict[str, float]] = {}

        for name, builder in model_builders.items():
            logger.info("\nTraining: %s", name)
            model = builder()

            # Cross-validation
            cv_metrics = self._cross_validate(model, X, y, name)
            all_results[name] = cv_metrics

            # Fit on full data for downstream SHAP analysis
            model.fit(X, y)
            full_metrics = self._evaluate(model, X, y, prefix="train")
            all_results[name].update(full_metrics)

            self.models[name] = model
            self.cv_results[name] = all_results[name]

        # Identify best model by mean CV R²
        self.best_model_name = max(
            all_results,
            key=lambda n: all_results[n].get("cv_r2_mean", -999),
        )
        self.best_model = self.models[self.best_model_name]

        logger.info("\n" + "=" * 60)
        logger.info("Best model: %s (CV R² = %.4f)",
                    self.best_model_name,
                    all_results[self.best_model_name]["cv_r2_mean"])
        logger.info("=" * 60)

        # Save results
        self._save_results(all_results)
        self._save_best_model()

        return all_results

    def _save_results(self, results: Dict[str, Dict[str, float]]) -> None:
        """Save a comparison table of model metrics to CSV."""
        if not self.output_dir:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for model_name, metrics in results.items():
            row = {"model": model_name}
            row.update(metrics)
            rows.append(row)

        df = pd.DataFrame(rows)
        out_path = self.output_dir / "model_comparison.csv"
        df.to_csv(out_path, index=False)
        logger.info("Model comparison table saved to %s", out_path)

        # Pretty-print to console
        print("\n" + "=" * 80)
        print("MODEL COMPARISON (5-Fold CV Results)")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)

    def _save_best_model(self) -> None:
        """Persist the best fitted model using joblib."""
        if not self.output_dir or self.best_model is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self.best_model_name.lower().replace(" ", "_")
        out_path = self.output_dir / f"best_model_{safe_name}.pkl"
        joblib.dump(self.best_model, out_path)
        logger.info("Best model saved to %s", out_path)

    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """
        Generate predictions using the specified model (default: best model).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (scaled).
        model_name : str, optional
            If None, uses the best model.

        Returns
        -------
        np.ndarray
            Predicted GHG emissions.
        """
        name = model_name or self.best_model_name
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found. Train models first.")
        return self.models[name].predict(X)

    def get_results_summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising all model CV results."""
        rows = [{"model": k, **v} for k, v in self.cv_results.items()]
        return pd.DataFrame(rows)
