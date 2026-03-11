"""
World Bank Data Fetcher Module

Fetches macroeconomic control variables from the World Bank API (wbgapi)
for 27 EU countries + Turkey, years 2014-2023.

Indicators fetched:
- NY.GDP.PCAP.KD   : GDP per capita (constant 2015 USD)
- EG.FEC.RNEW.ZS   : Renewable energy share in total final energy consumption (%)
- SP.URB.TOTL.IN.ZS: Urban population (% of total)
- BX.KLT.DINV.WD.GD.ZS: FDI net inflows (% of GDP)
- NV.IND.TOTL.ZS   : Industry (including construction) value added (% of GDP)
- NE.TRD.GNFS.ZS   : Trade (% of GDP)
"""

import logging
import warnings
from typing import Dict, List

import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TARGET_COUNTRIES = [
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "GR", "ES",
    "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
    "NL", "PL", "PT", "RO", "SE", "SI", "SK", "TR",
]

# Eurostat uses "EL" for Greece; World Bank uses "GR"
EUROSTAT_TO_WB_MAP: Dict[str, str] = {"EL": "GR"}
WB_TO_EUROSTAT_MAP: Dict[str, str] = {v: k for k, v in EUROSTAT_TO_WB_MAP.items()}

TARGET_YEARS = list(range(2014, 2024))

WB_INDICATORS: Dict[str, str] = {
    "NY.GDP.PCAP.KD": "gdp_per_capita",
    "EG.FEC.RNEW.ZS": "renewable_energy_share",
    "SP.URB.TOTL.IN.ZS": "urbanization_rate",
    "BX.KLT.DINV.WD.GD.ZS": "fdi_net_inflows",
    "NV.IND.TOTL.ZS": "industry_share",
    "NE.TRD.GNFS.ZS": "trade_share",
}


class WorldBankFetcher:
    """
    Fetches macroeconomic control variables from the World Bank Open Data API
    using the ``wbgapi`` library.
    """

    def __init__(self) -> None:
        self.countries = TARGET_COUNTRIES
        self.years = TARGET_YEARS
        self.indicators = WB_INDICATORS

    def _wb_country_codes(self) -> List[str]:
        """Return country codes mapped to World Bank conventions."""
        return [EUROSTAT_TO_WB_MAP.get(c, c) for c in self.countries]

    def _fetch_indicator(self, indicator_code: str, col_name: str) -> pd.DataFrame:
        """
        Fetch a single World Bank indicator for all target countries and years.

        Parameters
        ----------
        indicator_code : str
            World Bank indicator code (e.g. 'NY.GDP.PCAP.KD').
        col_name : str
            Column name to use in the returned DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: country (Eurostat code), year, <col_name>
        """
        try:
            import wbgapi as wb  # noqa: PLC0415

            wb_countries = self._wb_country_codes()
            raw = wb.data.DataFrame(
                indicator_code,
                economy=wb_countries,
                time=range(min(self.years), max(self.years) + 1),
                labels=False,
            )

            # wbgapi returns a DataFrame with economies as index and years as
            # columns (prefixed with "YR").  Melt to long format.
            raw = raw.reset_index()
            # Identify economy column
            eco_col = raw.columns[0]
            year_cols = [c for c in raw.columns if str(c).startswith("YR")]
            if not year_cols:
                # Fallback: try numeric column names
                year_cols = [c for c in raw.columns if str(c).isdigit()]

            df_long = raw.melt(
                id_vars=[eco_col], value_vars=year_cols,
                var_name="year_raw", value_name=col_name,
            )
            df_long.rename(columns={eco_col: "wb_country"}, inplace=True)
            df_long["year"] = (
                df_long["year_raw"].astype(str)
                .str.replace("YR", "", regex=False)
                .astype(int)
            )
            df_long.drop(columns=["year_raw"], inplace=True)

            # Map World Bank country codes back to Eurostat codes
            df_long["country"] = df_long["wb_country"].map(
                lambda c: WB_TO_EUROSTAT_MAP.get(c, c)
            )
            df_long.drop(columns=["wb_country"], inplace=True)

            df_long[col_name] = pd.to_numeric(df_long[col_name], errors="coerce")
            df_long = df_long[df_long["year"].isin(self.years)]
            df_long = df_long.dropna(subset=[col_name])

            return df_long[["country", "year", col_name]]

        except ImportError:
            logger.error("wbgapi is not installed. Run: pip install wbgapi")
            return pd.DataFrame(columns=["country", "year", col_name])
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            logger.warning("Failed to fetch WB indicator %s: %s", indicator_code, exc)
            return pd.DataFrame(columns=["country", "year", col_name])
        except Exception as exc:
            logger.warning(
                "Unexpected error fetching WB indicator %s: %s", indicator_code, exc
            )
            return pd.DataFrame(columns=["country", "year", col_name])

    def fetch_all(self) -> pd.DataFrame:
        """
        Fetch all macroeconomic indicators and merge into a single country-year
        panel DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: country, year, gdp_per_capita, renewable_energy_share,
                     urbanization_rate, fdi_net_inflows, industry_share, trade_share
        """
        logger.info("Fetching World Bank macroeconomic indicators...")

        frames = []
        for code, col in self.indicators.items():
            df = self._fetch_indicator(code, col)
            if not df.empty:
                frames.append(df)
                logger.info("  %s (%s): %d rows", col, code, len(df))
            else:
                logger.warning("  %s (%s): no data returned", col, code)

        if not frames:
            logger.warning("No World Bank data fetched.")
            return pd.DataFrame()

        # Merge all indicators on country + year
        result = frames[0]
        for df in frames[1:]:
            result = result.merge(df, on=["country", "year"], how="outer")

        # Keep only target countries (Eurostat codes) and years
        eurostat_targets = [
            WB_TO_EUROSTAT_MAP.get(c, c) for c in TARGET_COUNTRIES
        ]
        result = result[result["country"].isin(eurostat_targets)]
        result = result[result["year"].isin(self.years)]

        logger.info(
            "World Bank combined dataset: %d rows, %d columns.",
            len(result), result.shape[1],
        )
        return result
