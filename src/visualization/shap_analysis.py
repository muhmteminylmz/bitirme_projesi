"""
SHAP Analysis and Visualization Module

Uses the SHAP (SHapley Additive exPlanations) library to explain the
black-box Stacking Regressor and produce:
1. Feature Importance Bar Plot  (mean |SHAP| values)
2. SHAP Summary (Beeswarm) Plot (direction + magnitude of each feature)

Also produces supplementary visualizations:
3. Actual vs. Predicted scatter plot
4. Residuals distribution plot
5. Per-sector SHAP importance heatmap
"""

import logging
import warnings
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # Non-interactive backend for server/CI environments

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Plot aesthetics
FIGURE_DPI = 150
PALETTE = "viridis"
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


class SHAPAnalyzer:
    """
    Generates SHAP-based explanations and supplementary visualizations for the
    Twin Transition ML model.
    """

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shap_values: Optional[np.ndarray] = None
        self.shap_explainer = None

    # ------------------------------------------------------------------
    # SHAP computation
    # ------------------------------------------------------------------

    def compute_shap_values(
        self,
        model,
        X: pd.DataFrame,
        model_name: str = "best_model",
    ) -> Optional[np.ndarray]:
        """
        Compute SHAP values for the given model and feature matrix.

        For tree-based models (Random Forest, XGBoost) uses the fast
        TreeExplainer; for other models (Stacking, MLP) falls back to
        the model-agnostic KernelExplainer on a representative sample.

        Parameters
        ----------
        model : fitted sklearn / XGBoost estimator
        X : pd.DataFrame
            Feature matrix (scaled, without target column).
        model_name : str
            Used only for logging.

        Returns
        -------
        np.ndarray or None
            SHAP values array of shape (n_samples, n_features), or None
            if computation failed.
        """
        try:
            import shap  # noqa: PLC0415

            logger.info("Computing SHAP values for '%s'...", model_name)

            # Try TreeExplainer first (fast, exact for tree ensembles)
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                logger.info("Used TreeExplainer.")
            except (TypeError, AttributeError, ValueError, RuntimeError):
                # Fall back to KernelExplainer with a subsample
                sample_size = min(100, len(X))
                X_sample = shap.sample(X, sample_size, random_state=42)
                explainer = shap.KernelExplainer(model.predict, X_sample)
                shap_values = explainer.shap_values(X_sample)
                X = X_sample  # align X with shap_values
                logger.info("Used KernelExplainer on %d samples.", sample_size)

            self.shap_values = shap_values
            self.shap_explainer = explainer
            self._X_for_shap = X  # store for plots
            logger.info("SHAP values computed. Shape: %s", np.array(shap_values).shape)
            return shap_values

        except (ImportError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            if isinstance(exc, ImportError):
                logger.error("shap is not installed. Run: pip install shap")
            else:
                logger.error("SHAP computation failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # SHAP plots
    # ------------------------------------------------------------------

    def plot_feature_importance_bar(
        self,
        feature_names: List[str],
        shap_values: Optional[np.ndarray] = None,
        top_n: int = 15,
        save: bool = True,
    ) -> None:
        """
        Bar plot of mean absolute SHAP values (global feature importance).

        Parameters
        ----------
        feature_names : list of str
        shap_values : np.ndarray, optional
            If None, uses self.shap_values.
        top_n : int
            Number of top features to display.
        save : bool
            If True, saves the figure to output_dir.
        """
        sv = shap_values if shap_values is not None else self.shap_values
        if sv is None:
            logger.warning("No SHAP values available for bar plot.")
            return

        sv_arr = np.array(sv)
        mean_abs_shap = np.abs(sv_arr).mean(axis=0)
        importance_df = pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": mean_abs_shap}
        ).sort_values("mean_abs_shap", ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(
            importance_df["feature"][::-1],
            importance_df["mean_abs_shap"][::-1],
            color=plt.cm.viridis(  # type: ignore[attr-defined]
                np.linspace(0.2, 0.8, len(importance_df))
            ),
        )
        ax.set_xlabel("Mean |SHAP Value| (Impact on GHG Emissions)", fontsize=12)
        ax.set_title(
            "SHAP Feature Importance\n(Twin Transition: Digitalization → Carbon Emissions)",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.4)
        plt.tight_layout()

        if save:
            out_path = self.output_dir / "shap_feature_importance_bar.png"
            fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
            logger.info("Saved SHAP bar plot → %s", out_path)
        plt.close(fig)

    def plot_shap_summary(
        self,
        feature_names: List[str],
        shap_values: Optional[np.ndarray] = None,
        X: Optional[pd.DataFrame] = None,
        save: bool = True,
    ) -> None:
        """
        SHAP Summary (beeswarm) plot — shows direction and magnitude of each
        feature's impact on the target variable.

        Parameters
        ----------
        feature_names : list of str
        shap_values : np.ndarray, optional
        X : pd.DataFrame, optional
            Feature values (needed for color encoding).
        save : bool
        """
        sv = shap_values if shap_values is not None else self.shap_values
        X_plot = X if X is not None else getattr(self, "_X_for_shap", None)

        if sv is None:
            logger.warning("No SHAP values available for summary plot.")
            return

        try:
            import shap  # noqa: PLC0415

            fig, ax = plt.subplots(figsize=(11, 8))
            shap.summary_plot(
                sv,
                X_plot,
                feature_names=feature_names,
                show=False,
                plot_size=None,
            )
            plt.title(
                "SHAP Summary Plot\n(Twin Transition: Feature Impact on GHG Emissions)",
                fontsize=13,
                fontweight="bold",
            )
            plt.tight_layout()

            if save:
                out_path = self.output_dir / "shap_summary_plot.png"
                plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
                logger.info("Saved SHAP summary plot → %s", out_path)
            plt.close("all")

        except ImportError:
            logger.error("shap library not available for summary plot.")

    # ------------------------------------------------------------------
    # Supplementary visualizations
    # ------------------------------------------------------------------

    def plot_actual_vs_predicted(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_name: str = "Stacking Regressor",
        r2: Optional[float] = None,
        save: bool = True,
    ) -> None:
        """
        Scatter plot of actual vs. predicted GHG emissions.

        Parameters
        ----------
        y_true : pd.Series
            Actual GHG emissions.
        y_pred : np.ndarray
            Predicted GHG emissions.
        model_name : str
        r2 : float, optional
            R² score to display on the plot.
        save : bool
        """
        fig, ax = plt.subplots(figsize=(8, 7))

        ax.scatter(y_true, y_pred, alpha=0.5, s=20, color="#2980b9", edgecolors="none")

        # Perfect prediction line
        lims = [
            min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max()),
        ]
        ax.plot(lims, lims, "r--", lw=1.5, label="Perfect Prediction")

        title = f"Actual vs. Predicted GHG Emissions\n{model_name}"
        if r2 is not None:
            title += f"  |  R² = {r2:.4f}"
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Actual GHG Emissions (Thousand Tonnes CO₂-eq)", fontsize=11)
        ax.set_ylabel("Predicted GHG Emissions (Thousand Tonnes CO₂-eq)", fontsize=11)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save:
            safe_name = model_name.lower().replace(" ", "_")
            out_path = self.output_dir / f"actual_vs_predicted_{safe_name}.png"
            fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
            logger.info("Saved actual vs. predicted plot → %s", out_path)
        plt.close(fig)

    def plot_residuals(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_name: str = "Stacking Regressor",
        save: bool = True,
    ) -> None:
        """
        Residuals distribution plot (histogram + KDE).

        Parameters
        ----------
        y_true : pd.Series
        y_pred : np.ndarray
        model_name : str
        save : bool
        """
        residuals = np.array(y_true) - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram + KDE
        sns.histplot(residuals, kde=True, ax=axes[0], color="#27ae60", bins=40)
        axes[0].set_title(f"Residuals Distribution\n{model_name}", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Residual (Actual − Predicted)", fontsize=10)
        axes[0].axvline(0, color="red", linestyle="--", lw=1.5)

        # Residuals vs. Fitted
        axes[1].scatter(y_pred, residuals, alpha=0.4, s=15, color="#8e44ad")
        axes[1].axhline(0, color="red", linestyle="--", lw=1.5)
        axes[1].set_title(f"Residuals vs. Fitted\n{model_name}", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Fitted Values", fontsize=10)
        axes[1].set_ylabel("Residuals", fontsize=10)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()

        if save:
            safe_name = model_name.lower().replace(" ", "_")
            out_path = self.output_dir / f"residuals_{safe_name}.png"
            fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
            logger.info("Saved residuals plot → %s", out_path)
        plt.close(fig)

    def plot_model_comparison(
        self,
        results_df: pd.DataFrame,
        save: bool = True,
    ) -> None:
        """
        Bar chart comparing R² scores across all trained models.

        Parameters
        ----------
        results_df : pd.DataFrame
            Output of ModelTrainer.get_results_summary().
        save : bool
        """
        if "cv_r2_mean" not in results_df.columns:
            logger.warning("results_df missing 'cv_r2_mean' column.")
            return

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
        bars = ax.bar(
            results_df["model"],
            results_df["cv_r2_mean"],
            yerr=results_df.get("cv_r2_std", None),
            capsize=5,
            color=colors[: len(results_df)],
            alpha=0.85,
            edgecolor="white",
        )

        ax.set_title(
            "Model Comparison: 5-Fold CV R² Scores\n"
            "(Twin Transition — GHG Emissions Prediction)",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_ylabel("Cross-Validated R²", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.9, color="black", linestyle="--", lw=1, label="Target R²=0.90")
        ax.legend(fontsize=9)

        for bar, val in zip(bars, results_df["cv_r2_mean"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()

        if save:
            out_path = self.output_dir / "model_comparison_r2.png"
            fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
            logger.info("Saved model comparison plot → %s", out_path)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Full analysis pipeline
    # ------------------------------------------------------------------

    def run_full_analysis(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str],
        model_name: str = "Stacking Regressor",
        results_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Run the complete SHAP + visualization pipeline.

        Parameters
        ----------
        model : fitted estimator
        X : pd.DataFrame
            Scaled feature matrix.
        y : pd.Series
            True target values.
        feature_names : list of str
        model_name : str
        results_df : pd.DataFrame, optional
            Model comparison results for the comparison bar chart.
        """
        logger.info("=" * 60)
        logger.info("Starting SHAP and visualization analysis for '%s'.", model_name)
        logger.info("=" * 60)

        # 1. Compute SHAP values
        shap_values = self.compute_shap_values(model, X, model_name)

        # 2. SHAP Feature Importance Bar Plot
        if shap_values is not None:
            self.plot_feature_importance_bar(feature_names, shap_values)
            self.plot_shap_summary(feature_names, shap_values, X)

        # 3. Actual vs. Predicted
        from sklearn.metrics import r2_score  # noqa: PLC0415
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        self.plot_actual_vs_predicted(y, y_pred, model_name, r2)

        # 4. Residuals
        self.plot_residuals(y, y_pred, model_name)

        # 5. Model comparison (if results provided)
        if results_df is not None and not results_df.empty:
            self.plot_model_comparison(results_df)

        logger.info("All plots saved to: %s", self.output_dir)
