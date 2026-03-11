"""
Eurostat Data Fetcher Module

Fetches sectoral GHG emissions and digitalization indicators from Eurostat
for 27 EU countries + Turkey (28 total), years 2014-2023,
across NACE Rev. 2 sectors: C (Manufacturing), F (Construction),
G (Trade), H (Logistics), J (ICT).
"""

import logging
import warnings
from typing import Optional

import pandas as pd
import requests

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

EUROSTAT_BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"


class EurostatFetcher:
    """
    Fetches and processes Eurostat datasets relevant to the Twin Transition project.

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

    def _build_params(self, dataset: str, extra_params: dict) -> dict:
        """Build common query parameters for the Eurostat JSON API."""
        params = {
            "format": "JSON",
            "lang": "EN",
            "geo": self.countries,
            "time": [str(y) for y in self.years],
        }
        params.update(extra_params)
        return params

    def _fetch_json(self, dataset: str, params: dict) -> Optional[dict]:
        """Low-level GET request to Eurostat JSON API."""
        url = f"{EUROSTAT_BASE_URL}/{dataset}"
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            logger.warning("Failed to fetch %s: %s", dataset, exc)
            return None

    @staticmethod
    def _json_to_dataframe(data: dict, value_col: str) -> pd.DataFrame:
        """
        Convert a Eurostat JSON-stat response to a tidy DataFrame with
        columns: country, year, sector (if applicable), <value_col>.
        """
        if data is None:
            return pd.DataFrame()

        try:
            dims = data["dimension"]
            dim_ids = data["id"]
            values = data["value"]

            # Build index label maps
            label_maps = {}
            for dim_name in dim_ids:
                label_maps[dim_name] = {
                    int(v): k
                    for k, v in dims[dim_name]["category"]["index"].items()
                }

            # Dimension sizes for position calculation
            sizes = [len(label_maps[d]) for d in dim_ids]

            records = []
            for flat_idx_str, val in values.items():
                flat_idx = int(flat_idx_str)
                coords = {}
                remaining = flat_idx
                for i, dim_name in enumerate(reversed(dim_ids)):
                    size = sizes[len(dim_ids) - 1 - i]
                    coords[dim_name] = label_maps[dim_name][remaining % size]
                    remaining //= size

                record = {dim: coords[dim] for dim in dim_ids}
                record[value_col] = val
                records.append(record)

            df = pd.DataFrame(records)
            if df.empty:
                return df

            # Rename standard Eurostat dimension names
            rename_map = {}
            for col in df.columns:
                if col.lower() in ("geo", "geo\\time"):
                    rename_map[col] = "country"
                elif col.lower() == "time":
                    rename_map[col] = "year"
                elif col.lower() in ("nace_r2",):
                    rename_map[col] = "sector"
            df.rename(columns=rename_map, inplace=True)

            if "year" in df.columns:
                df["year"] = pd.to_numeric(df["year"], errors="coerce")
            return df

        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Could not parse JSON response: %s", exc)
            return pd.DataFrame()

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

        # Map our target sectors to the Eurostat NACE Rev. 2 codes used in
        # env_ac_ainah_r2 (e.g. NACE_R2_C, NACE_R2_F, etc.)
        nace_codes = [f"NACE_R2_{s}" for s in self.sectors.keys()]

        params = self._build_params(
            "env_ac_ainah_r2",
            {
                "airpol": "GHG",
                "unit": "THS_T",
                "nace_r2": nace_codes,
            },
        )
        data = self._fetch_json("env_ac_ainah_r2", params)
        df = self._json_to_dataframe(data, "ghg_emissions")

        if not df.empty and "sector" in df.columns:
            # Strip the NACE_R2_ prefix to keep single-letter codes
            df["sector"] = df["sector"].str.replace("NACE_R2_", "", regex=False)
            df = df[df["sector"].isin(self.sectors.keys())]

        if not df.empty:
            df = df[df["country"].isin(self.countries)]
            df = df[df["year"].isin(self.years)]
            df = df.dropna(subset=["ghg_emissions"])
            df["ghg_emissions"] = pd.to_numeric(df["ghg_emissions"], errors="coerce")

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

        params = self._build_params(
            "isoc_eb_iip",
            {
                "indic_is": "E_ERPEUSO",
                "unit": "PC_ENT",
                "sizen_r2": "10_C10_S951_XK",
                "nace_r2": list(self.sectors.keys()),
            },
        )
        data = self._fetch_json("isoc_eb_iip", params)
        df = self._json_to_dataframe(data, "erp_usage")

        if not df.empty:
            df = df[df["country"].isin(self.countries)]
            df = df[df["year"].isin(self.years)]
            df = df.dropna(subset=["erp_usage"])
            df["erp_usage"] = pd.to_numeric(df["erp_usage"], errors="coerce")

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

        params = self._build_params(
            "isoc_cicce_use",
            {
                "indic_is": "E_CC",
                "unit": "PC_ENT",
                "sizen_r2": "10_C10_S951_XK",
                "nace_r2": list(self.sectors.keys()),
            },
        )
        data = self._fetch_json("isoc_cicce_use", params)
        df = self._json_to_dataframe(data, "cloud_usage")

        if not df.empty:
            df = df[df["country"].isin(self.countries)]
            df = df[df["year"].isin(self.years)]
            df = df.dropna(subset=["cloud_usage"])
            df["cloud_usage"] = pd.to_numeric(df["cloud_usage"], errors="coerce")

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

        params = self._build_params(
            "isoc_sks_itspt",
            {
                "indic_is": "ISS_TOTEMP",
                "unit": "PC_EMP",
            },
        )
        data = self._fetch_json("isoc_sks_itspt", params)
        df = self._json_to_dataframe(data, "ict_employment")

        if not df.empty:
            df = df[df["country"].isin(self.countries)]
            df = df[df["year"].isin(self.years)]
            df = df.dropna(subset=["ict_employment"])
            df["ict_employment"] = pd.to_numeric(df["ict_employment"], errors="coerce")

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
