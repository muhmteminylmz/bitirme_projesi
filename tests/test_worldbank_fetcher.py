"""
Tests for the WorldBankFetcher class.

Uses mocked ``wbgapi`` to validate data fetching, error handling, and
country code mapping without requiring internet access.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from requests.exceptions import JSONDecodeError as RequestsJSONDecodeError

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.worldbank_fetcher import (
    WorldBankFetcher,
    TARGET_COUNTRIES,
    WB_INDICATORS,
    EUROSTAT_TO_WB_MAP,
    WB_TO_EUROSTAT_MAP,
)


# ---------------------------------------------------------------------------
# Helpers to build mock wbgapi DataFrames
# ---------------------------------------------------------------------------

def _make_wb_dataframe(countries, years, value=1000.0):
    """Build a mock DataFrame resembling wbgapi.data.DataFrame output.

    wbgapi returns economies as the index and year columns prefixed with 'YR'.
    """
    data = {"economy": countries}
    for year in years:
        data[f"YR{year}"] = [value + i for i in range(len(countries))]
    return pd.DataFrame(data).set_index("economy")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWorldBankFetcher:

    def test_wb_country_codes_maps_el_to_gr(self):
        """Eurostat 'EL' for Greece must map to 'GR' for World Bank."""
        fetcher = WorldBankFetcher()
        wb_codes = fetcher._wb_country_codes()
        assert "GR" in wb_codes
        assert "EL" not in wb_codes

    def test_fetch_indicator_returns_dataframe(self):
        """A successful API call should return a non-empty DataFrame."""
        mock_raw = _make_wb_dataframe(["DE", "FR", "GR"], [2020, 2021])
        mock_wb = MagicMock()
        mock_wb.data.DataFrame.return_value = mock_raw

        fetcher = WorldBankFetcher()
        with patch.dict("sys.modules", {"wbgapi": mock_wb}):
            result = fetcher._fetch_indicator("NY.GDP.PCAP.KD", "gdp_per_capita")

        assert not result.empty
        assert "country" in result.columns
        assert "year" in result.columns
        assert "gdp_per_capita" in result.columns

    def test_fetch_indicator_maps_gr_back_to_el(self):
        """World Bank 'GR' must be mapped back to Eurostat 'EL'."""
        mock_raw = _make_wb_dataframe(["GR"], [2020])
        mock_wb = MagicMock()
        mock_wb.data.DataFrame.return_value = mock_raw

        fetcher = WorldBankFetcher()
        with patch.dict("sys.modules", {"wbgapi": mock_wb}):
            result = fetcher._fetch_indicator("NY.GDP.PCAP.KD", "gdp_per_capita")

        assert "EL" in result["country"].values
        assert "GR" not in result["country"].values

    def test_json_decode_error_returns_empty(self):
        """A JSONDecodeError from the API should be caught gracefully."""
        mock_wb = MagicMock()
        mock_wb.data.DataFrame.side_effect = json.JSONDecodeError("Expecting value", "", 0)

        fetcher = WorldBankFetcher()
        with patch.dict("sys.modules", {"wbgapi": mock_wb}):
            result = fetcher._fetch_indicator("NY.GDP.PCAP.KD", "gdp_per_capita")

        assert result.empty
        assert list(result.columns) == ["country", "year", "gdp_per_capita"]

    def test_requests_json_decode_error_returns_empty(self):
        """A requests.exceptions.JSONDecodeError should be caught gracefully."""
        mock_wb = MagicMock()
        mock_wb.data.DataFrame.side_effect = RequestsJSONDecodeError(
            "Expecting value", "", 0
        )

        fetcher = WorldBankFetcher()
        with patch.dict("sys.modules", {"wbgapi": mock_wb}):
            result = fetcher._fetch_indicator("NY.GDP.PCAP.KD", "gdp_per_capita")

        assert result.empty
        assert list(result.columns) == ["country", "year", "gdp_per_capita"]

    def test_runtime_error_returns_empty(self):
        """A RuntimeError from wbgapi internals should be caught gracefully."""
        mock_wb = MagicMock()
        mock_wb.data.DataFrame.side_effect = RuntimeError("API unavailable")

        fetcher = WorldBankFetcher()
        with patch.dict("sys.modules", {"wbgapi": mock_wb}):
            result = fetcher._fetch_indicator("NY.GDP.PCAP.KD", "gdp_per_capita")

        assert result.empty
        assert list(result.columns) == ["country", "year", "gdp_per_capita"]

    def test_connection_error_returns_empty(self):
        """A ConnectionError should be caught gracefully."""
        mock_wb = MagicMock()
        mock_wb.data.DataFrame.side_effect = ConnectionError("No internet")

        fetcher = WorldBankFetcher()
        with patch.dict("sys.modules", {"wbgapi": mock_wb}):
            result = fetcher._fetch_indicator("NY.GDP.PCAP.KD", "gdp_per_capita")

        assert result.empty

    def test_fetch_all_returns_empty_when_all_indicators_fail(self):
        """If every indicator fails, fetch_all should return an empty DataFrame."""
        mock_wb = MagicMock()
        mock_wb.data.DataFrame.side_effect = RuntimeError("API down")

        fetcher = WorldBankFetcher()
        with patch.dict("sys.modules", {"wbgapi": mock_wb}):
            result = fetcher.fetch_all()

        assert result.empty

    def test_fetch_all_merges_successful_indicators(self):
        """fetch_all should merge data from all indicators that succeed."""
        mock_raw = _make_wb_dataframe(["DE", "FR"], [2020, 2021])
        mock_wb = MagicMock()
        mock_wb.data.DataFrame.return_value = mock_raw

        fetcher = WorldBankFetcher()
        with patch.dict("sys.modules", {"wbgapi": mock_wb}):
            result = fetcher.fetch_all()

        assert not result.empty
        assert "country" in result.columns
        assert "year" in result.columns
