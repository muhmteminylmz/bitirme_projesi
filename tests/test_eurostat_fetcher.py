"""
Tests for the EurostatFetcher class.

Uses mocked ``eurostat.get_data_df`` to validate filtering, NACE code
mapping, and data reshaping without requiring internet access.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.eurostat_fetcher import EurostatFetcher, ISOC_NACE_MAPPING


# ---------------------------------------------------------------------------
# Helpers to build mock Eurostat DataFrames
# ---------------------------------------------------------------------------

def _make_isoc_df(nace_codes, indicator_col, indicator_val, value=42.0):
    """Build a mock DataFrame resembling an ISOC digital dataset."""
    rows = []
    for geo in ["DE", "FR", "TR"]:
        for nace in nace_codes:
            rows.append({
                "indic_is": indicator_val,
                "unit": "PC_ENT",
                "sizen_r2": "10_C10_S951_XK",
                "nace_r2": nace,
                "geo": geo,
                "2020": value,
                "2021": value + 1,
            })
    return pd.DataFrame(rows)


def _make_ghg_df():
    """Build a mock DataFrame resembling env_ac_ainah_r2."""
    rows = []
    for geo in ["DE", "FR", "TR"]:
        for nace in ["C", "F", "G", "H", "J"]:
            rows.append({
                "airpol": "GHG",
                "unit": "THS_T",
                "nace_r2": nace,
                "geo": geo,
                "2020": 1000.0,
                "2021": 1100.0,
            })
    return pd.DataFrame(rows)


def _make_ict_df():
    """Build a mock DataFrame resembling isoc_sks_itspt."""
    rows = []
    for geo in ["DE", "FR", "TR"]:
        rows.append({
            "unit": "PC_EMP",
            "sex": "T",
            "age": "Y15-74",
            "geo": geo,
            "2020": 5.0,
            "2021": 5.5,
        })
        # Extra row that should be filtered out (wrong sex)
        rows.append({
            "unit": "PC_EMP",
            "sex": "M",
            "age": "Y15-74",
            "geo": geo,
            "2020": 7.0,
            "2021": 7.5,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEurostatFetcher:

    def test_isoc_nace_mapping_c10_c33_to_c(self):
        """ISOC datasets code C10-C33 must be mapped to standard sector C."""
        mock_df = _make_isoc_df(
            nace_codes=["C10-C33", "F", "G", "H", "J"],
            indicator_col="indic_is",
            indicator_val="E_ERP1",
        )
        fetcher = EurostatFetcher()
        with patch("src.data_collection.eurostat_fetcher.eurostat.get_data_df", return_value=mock_df):
            result = fetcher.fetch_erp_usage()

        assert not result.empty
        assert "C" in result["sector"].values
        assert "C10-C33" not in result["sector"].values

    def test_erp_uses_correct_indicator_code(self):
        """ERP fetcher must filter on indic_is=E_ERP1."""
        mock_df = _make_isoc_df(
            nace_codes=["C10-C33", "F"],
            indicator_col="indic_is",
            indicator_val="E_ERP1",
        )
        # Add rows with a wrong indicator that should be filtered out
        extra = mock_df.copy()
        extra["indic_is"] = "E_WRONG"
        mock_df = pd.concat([mock_df, extra], ignore_index=True)

        fetcher = EurostatFetcher()
        with patch("src.data_collection.eurostat_fetcher.eurostat.get_data_df", return_value=mock_df):
            result = fetcher.fetch_erp_usage()

        # Should only have rows from E_ERP1
        assert not result.empty
        assert len(result) == 3 * 2 * 2  # 3 countries × 2 sectors × 2 years

    def test_ict_filters_sex_and_age(self):
        """ICT employment must filter sex=T and age=Y15-74."""
        mock_df = _make_ict_df()
        fetcher = EurostatFetcher()
        with patch("src.data_collection.eurostat_fetcher.eurostat.get_data_df", return_value=mock_df):
            result = fetcher.fetch_ict_employment()

        assert not result.empty
        # Only sex=T rows should remain (3 countries × 2 years)
        assert len(result) == 3 * 2

    def test_ghg_uses_standard_nace_codes(self):
        """GHG fetcher must use standard NACE codes (C, F, G, H, J) directly."""
        mock_df = _make_ghg_df()
        fetcher = EurostatFetcher()
        with patch("src.data_collection.eurostat_fetcher.eurostat.get_data_df", return_value=mock_df):
            result = fetcher.fetch_ghg_emissions()

        assert not result.empty
        assert set(result["sector"].unique()) == {"C", "F", "G", "H", "J"}

    def test_cloud_uses_isoc_mapping(self):
        """Cloud fetcher must use ISOC NACE mapping like ERP."""
        mock_df = _make_isoc_df(
            nace_codes=["C10-C33", "J"],
            indicator_col="indic_is",
            indicator_val="E_CC",
        )
        fetcher = EurostatFetcher()
        with patch("src.data_collection.eurostat_fetcher.eurostat.get_data_df", return_value=mock_df):
            result = fetcher.fetch_cloud_usage()

        assert not result.empty
        assert "C" in result["sector"].values

    def test_fetch_all_merges_correctly(self):
        """fetch_all should merge GHG, ERP, cloud, and ICT data."""
        ghg_df = _make_ghg_df()
        erp_df = _make_isoc_df(["C10-C33", "F", "G", "H", "J"], "indic_is", "E_ERP1")
        cloud_df = _make_isoc_df(["C10-C33", "F", "G", "H", "J"], "indic_is", "E_CC", value=30.0)
        ict_df = _make_ict_df()

        call_count = {"n": 0}
        datasets = [ghg_df, erp_df, cloud_df, ict_df]

        def mock_get_data_df(code):
            df = datasets[call_count["n"]]
            call_count["n"] += 1
            return df

        fetcher = EurostatFetcher()
        with patch("src.data_collection.eurostat_fetcher.eurostat.get_data_df", side_effect=mock_get_data_df):
            result = fetcher.fetch_all()

        assert not result.empty
        expected_cols = {"country", "year", "sector", "ghg_emissions",
                         "erp_usage", "cloud_usage", "ict_employment"}
        assert expected_cols.issubset(set(result.columns))

    def test_empty_api_response_returns_empty(self):
        """If the API returns None, fetcher should return an empty DataFrame."""
        fetcher = EurostatFetcher()
        with patch("src.data_collection.eurostat_fetcher.eurostat.get_data_df", return_value=None):
            result = fetcher.fetch_ghg_emissions()

        assert result.empty

    def test_api_exception_returns_empty(self):
        """If the API raises an exception, fetcher should return an empty DataFrame."""
        fetcher = EurostatFetcher()
        with patch("src.data_collection.eurostat_fetcher.eurostat.get_data_df", side_effect=Exception("API error")):
            result = fetcher.fetch_erp_usage()

        assert result.empty
