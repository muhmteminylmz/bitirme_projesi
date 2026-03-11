"""
Eurostat Data Fetcher Module

Fetches sectoral GHG emissions and digitalization indicators from Eurostat
using the ``eurostat`` Python library for 27 EU countries + Turkey (28 total),
years 2014-2023, across NACE Rev. 2 sectors: C (Manufacturing),
F (Construction), G (Trade), H (Logistics), J (ICT).
"""

import logging
import warnings
from typing import Dict, List, Optional

import eurostat
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 27 EU member states + Turkey
TARGET_COUNTRIES = [
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES",
    "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
    "NL", "PL", "PT", "RO", "SE", "SI", "SK", "TR",
]

TARGET_YEARS = list(range(2014, 2024))  # 2014 - 2023 inclusive

# NACE Rev. 2 target sectors
TARGET_SECTORS = {
    "C": "Manufacturing",
    "F": "Construction",
    "G": "Trade",
    "H": "Logistics",
    "J": "ICT",
}

# ISOC digital datasets use extended NACE codes; map them to standard ones
ISOC_NACE_MAPPING: Dict[str, str] = {
    "C10-C33": "C",
    "F": "F",
    "G": "G",
    "H": "H",
    "J": "J",
}


class EurostatFetcher:
    """
    Fetches and processes Eurostat datasets relevant to the Twin Transition project.

    Uses the ``eurostat`` Python library (``eurostat.get_data_df``) to pull
    bulk data and then filters/reshapes it locally.

    Datasets used:
    - env_ac_ainah_r2 : Sectoral GHG / air emissions by NACE Rev. 2 activity
    - isoc_eb_iip     : Enterprises using ERP systems
    - isoc_cicce_use  : Enterprises using cloud computing
    - isoc_sks_itspt  : ICT specialist employment share
    """

    def __init__(self) -> None:
        self.countries = TARGET_COUNTRIES
        self.years = TARGET_YEARS
        self.sectors = TARGET_SECTORS

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_and_filter(
        self,
        dataset_code: str,
        value_col: str,
        filter_dict: Dict[str, object],
        is_isoc: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch a Eurostat dataset via the ``eurostat`` library, apply column
        filters, map NACE sector codes, and melt into long format.

        Parameters
        ----------
        dataset_code : str
            Eurostat dataset identifier (e.g. ``'env_ac_ainah_r2'``).
        value_col : str
            Name for the value column in the returned DataFrame.
        filter_dict : dict
            Column-name → value(s) filters to apply.  Values may be a single
            string or a list of strings.
        is_isoc : bool
            If True, treat NACE codes as ISOC-style (e.g. ``C10-C33``) and
            map them to standard single-letter codes.

        Returns
        -------
        pd.DataFrame
            Tidy long-format DataFrame with columns including ``country``,
            ``year``, optionally ``sector``, and ``<value_col>``.
        """
        try:
            df = eurostat.get_data_df(dataset_code)
        except Exception as exc:
            logger.warning("Failed to fetch %s via eurostat library: %s", dataset_code, exc)
            return pd.DataFrame()

        if df is None or df.empty:
            logger.warning("No data returned for %s.", dataset_code)
            return pd.DataFrame()

        # Normalise column names (some versions append \\TIME_PERIOD)
        df.columns = [str(col).replace("\\TIME_PERIOD", "").strip() for col in df.columns]

        # Apply filters
        for col, val in filter_dict.items():
            if col in df.columns:
                if isinstance(val, list):
                    df = df[df[col].isin(val)]
                else:
                    df = df[df[col] == val]

        # Filter to target countries
        if "geo" in df.columns:
            df = df[df["geo"].isin(self.countries)].copy()

        # NACE sector filtering and mapping
        has_sector = "nace_r2" in df.columns
        if has_sector:
            if is_isoc:
                df = df[df["nace_r2"].isin(ISOC_NACE_MAPPING.keys())].copy()
                df["nace_r2"] = df["nace_r2"].map(ISOC_NACE_MAPPING)
            else:
                df = df[df["nace_r2"].isin(self.sectors.keys())].copy()

        # Identify year columns and melt to long format
        year_cols = [str(y) for y in self.years if str(y) in df.columns]
        if not year_cols:
            logger.warning("No year columns found in %s.", dataset_code)
            return pd.DataFrame()

        id_vars: List[str] = ["geo"]
        if has_sector:
            id_vars.append("nace_r2")

        df = df.melt(id_vars=id_vars, value_vars=year_cols, var_name="year", value_name=value_col)
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

        # Rename to standard column names
        df.rename(columns={"geo": "country"}, inplace=True)
        if has_sector:
            df.rename(columns={"nace_r2": "sector"}, inplace=True)

        # Clean values
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[value_col])

        # Aggregate in case of duplicate entries
        group_cols = ["country", "year"]
        if has_sector:
            group_cols.append("sector")
        df = df.groupby(group_cols, as_index=False)[value_col].mean()

        return df

    # ------------------------------------------------------------------
    # Individual dataset fetchers
    # ------------------------------------------------------------------

    def fetch_ghg_emissions(self) -> pd.DataFrame:
        """
        Fetch sectoral GHG emissions (CO2 equivalent, thousand tonnes) from
        env_ac_ainah_r2. Filtered to airpol=GHG and unit=THS_T.

        Returns
        -------
        pd.DataFrame
            Columns: country, year, sector, ghg_emissions
        """
        logger.info("Fetching GHG emissions data (env_ac_ainah_r2)...")

        df = self._fetch_and_filter(
            dataset_code="env_ac_ainah_r2",
            value_col="ghg_emissions",
            filter_dict={"airpol": "GHG", "unit": "THS_T"},
            is_isoc=False,
        )

        logger.info("GHG emissions: %d rows fetched.", len(df))
        return df

    def fetch_erp_usage(self) -> pd.DataFrame:
        """
        Fetch ERP software usage rate (% of enterprises) from isoc_eb_iip.

        Returns
        -------
        pd.DataFrame
            Columns: country, year, sector, erp_usage
        """
        logger.info("Fetching ERP usage data (isoc_eb_iip)...")

        df = self._fetch_and_filter(
            dataset_code="isoc_eb_iip",
            value_col="erp_usage",
            filter_dict={
                "indic_is": "E_ERP1",
                "unit": "PC_ENT",
                "sizen_r2": "10_C10_S951_XK",
            },
            is_isoc=True,
        )

        logger.info("ERP usage: %d rows fetched.", len(df))
        return df

    def fetch_cloud_usage(self) -> pd.DataFrame:
        """
        Fetch cloud computing usage rate (% of enterprises) from isoc_cicce_use.

        Returns
        -------
        pd.DataFrame
            Columns: country, year, sector, cloud_usage
        """
        logger.info("Fetching cloud computing usage data (isoc_cicce_use)...")

        df = self._fetch_and_filter(
            dataset_code="isoc_cicce_use",
            value_col="cloud_usage",
            filter_dict={
                "indic_is": "E_CC",
                "unit": "PC_ENT",
                "sizen_r2": "10_C10_S951_XK",
            },
            is_isoc=True,
        )

        logger.info("Cloud usage: %d rows fetched.", len(df))
        return df

    def fetch_ict_employment(self) -> pd.DataFrame:
        """
        Fetch ICT specialist employment share (% of total employment) from
        isoc_sks_itspt.

        Returns
        -------
        pd.DataFrame
            Columns: country, year, ict_employment
        """
        logger.info("Fetching ICT specialist employment data (isoc_sks_itspt)...")

        df = self._fetch_and_filter(
            dataset_code="isoc_sks_itspt",
            value_col="ict_employment",
            filter_dict={
                "unit": "PC_EMP",
                "sex": "T",
                "age": "Y15-74",
            },
            is_isoc=False,
        )

        logger.info("ICT employment: %d rows fetched.", len(df))
        return df

    # ------------------------------------------------------------------
    # Combined fetcher
    # ------------------------------------------------------------------

    def fetch_all(self) -> pd.DataFrame:
        """
        Fetch and merge all Eurostat indicators into a single panel DataFrame.

        Returns
        -------
        pd.DataFrame
            Merged panel data indexed by (country, year, sector).
        """
        ghg = self.fetch_ghg_emissions()
        erp = self.fetch_erp_usage()
        cloud = self.fetch_cloud_usage()
        ict = self.fetch_ict_employment()

        # Sector-level merge base
        if ghg.empty:
            logger.warning("GHG emissions data is empty. Returning empty DataFrame.")
            return pd.DataFrame()

        df = ghg.copy()

        # Merge ERP (sector-level)
        if not erp.empty and "sector" in erp.columns:
            merge_cols = ["country", "year", "sector"]
            erp_cols = merge_cols + ["erp_usage"]
            df = df.merge(erp[erp_cols], on=merge_cols, how="left")
        else:
            df["erp_usage"] = float("nan")

        # Merge cloud (sector-level)
        if not cloud.empty and "sector" in cloud.columns:
            merge_cols = ["country", "year", "sector"]
            cloud_cols = merge_cols + ["cloud_usage"]
            df = df.merge(cloud[cloud_cols], on=merge_cols, how="left")
        else:
            df["cloud_usage"] = float("nan")

        # ICT employment is country-year level (not sector-level)
        if not ict.empty:
            ict_cols = ["country", "year", "ict_employment"]
            df = df.merge(ict[ict_cols], on=["country", "year"], how="left")
        else:
            df["ict_employment"] = float("nan")

        logger.info("Eurostat combined dataset: %d rows.", len(df))
        return df
