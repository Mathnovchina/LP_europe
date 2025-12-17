"""Build the 1995-2022 macro panel directly from Eurostat."""
from __future__ import annotations

import io
import re
from functools import reduce
from pathlib import Path

import pandas as pd
import xlrd
from eurostat import get_data_df
import requests

COUNTRIES = [
    "AT",
    "BE",
    "BG",
    "CY",
    "CZ",
    "DE",
    "DK",
    "EE",
    "EL",
    "ES",
    "FI",
    "FR",
    "HR",
    "HU",
    "IE",
    "IT",
    "LT",
    "LU",
    "LV",
    "CH",
    "MT",
    "NL",
    "NO",
    "PL",
    "PT",
    "RO",
    "SE",
    "SI",
    "SK",
    "UK",
    "IS",
]
YEARS = range(1995, 2023)
ANNUAL_TIMES = [str(year) for year in YEARS]
MONTHLY_TIMES = [f"{year}M{month:02d}" for year in YEARS for month in range(1, 13)]
TIME_COL_PATTERN = re.compile(r"^\d{4}")

COFOG_TOTAL_CODES = [f"GF{i:02d}" for i in range(1, 11)]

ISO2_TO_ISO3 = {
    "AT": "AUT",
    "BE": "BEL",
    "BG": "BGR",
    "CY": "CYP",
    "CZ": "CZE",
    "DE": "DEU",
    "DK": "DNK",
    "EE": "EST",
    "EL": "GRC",
    "ES": "ESP",
    "FI": "FIN",
    "FR": "FRA",
    "HR": "HRV",
    "HU": "HUN",
    "IE": "IRL",
    "IT": "ITA",
    "LT": "LTU",
    "LU": "LUX",
    "LV": "LVA",
    "CH": "CHE",
    "MT": "MLT",
    "NL": "NLD",
    "NO": "NOR",
    "PL": "POL",
    "PT": "PRT",
    "RO": "ROU",
    "SE": "SWE",
    "SI": "SVN",
    "SK": "SVK",
    "UK": "GBR",
    "IS": "ISL",
}

TABLES = {
    "gdp_nominal": {
        "code": "nama_10_gdp",
        "filters": {"freq": "A", "na_item": "B1GQ", "unit": "CP_MEUR", "time": ANNUAL_TIMES},
    },
    "gdp_const_2015": {
        "code": "nama_10_gdp",
        "filters": {"freq": "A", "na_item": "B1GQ", "unit": "CLV15_MEUR", "time": ANNUAL_TIMES},
    },
    "gfcf_total": {
        "code": "nama_10_gdp",
        "filters": {"freq": "A", "na_item": "P51G", "unit": "CP_MEUR", "time": ANNUAL_TIMES},
    },
    "gfcf_public": {
        "code": "nasa_10_nf_tr",
        "filters": {
            "freq": "A",
            "na_item": "P51G",
            "unit": "CP_MEUR",
            "sector": "S13",
            "direct": "PAID",
            "time": ANNUAL_TIMES,
        },
    },
    "env_consumption": {
        "code": "gov_10a_exp",
        "filters": {
            "freq": "A",
            "cofog99": "GF05",
            "sector": "S13",
            "na_item": "P3",
            "unit": "MIO_EUR",
            "time": ANNUAL_TIMES,
        },
    },
    "env_investment": {
        "code": "gov_10a_exp",
        "filters": {
            "freq": "A",
            "cofog99": "GF05",
            "sector": "S13",
            "na_item": "P51G",
            "unit": "MIO_EUR",
            "time": ANNUAL_TIMES,
        },
    },
    "population": {
        "code": "demo_gind",
        "filters": {
            "freq": "A",
            "indic_de": "JAN",
            "sex": "T",
            "age": "TOTAL",
            "time": ANNUAL_TIMES,
        },
    },
    "real_gdp_per_capita": {
        "code": "nama_10_pc",
        "filters": {
            "freq": "A",
            "na_item": "B1GQ",
            "unit": "CLV15_EUR_HAB",
            "time": ANNUAL_TIMES,
        },
    },
    "workers_thousands": {
        "code": "nama_10_a10_e",
        "filters": {
            "freq": "A",
            "nace_r2": "TOTAL",
            "na_item": "EMP_DC",
            "unit": "THS_PER",
            "time": ANNUAL_TIMES,
        },
    },
    "hours_worked_thousands": {
        "code": "nama_10_a10_e",
        "filters": {
            "freq": "A",
            "nace_r2": "TOTAL",
            "na_item": "EMP_DC",
            "unit": "THS_HW",
            "time": ANNUAL_TIMES,
        },
    },
    "value_added": {
        "code": "nama_10_a10",
        "filters": {
            "freq": "A",
            "nace_r2": "TOTAL",
            "na_item": "B1G",
            "unit": "CP_MEUR",
            "time": ANNUAL_TIMES,
        },
    },
    "short_rate": {
        "code": "ei_mfir_m",
        "filters": {
            "freq": "M",
            "s_adj": "NSA",
            "p_adj": "NAP",
            "indic": "MF-3MI-RT",
            "time": MONTHLY_TIMES,
        },
        "annual_mean": True,
    },
    "long_rate": {
        "code": "ei_mfir_m",
        "filters": {
            "freq": "M",
            "s_adj": "NSA",
            "p_adj": "NAP",
            "indic": "MF-LTGBY-RT",
            "time": MONTHLY_TIMES,
        },
        "annual_mean": True,
    },
    "long_rate_ecb": {
        "code": "irt_lt_mcby_a",
        "filters": {
            "freq": "A",
            "int_rt": "MCBY",
            "time": ANNUAL_TIMES,
        },
    },
}

# Source https://www.indexmundi.com/facts/malta/indicator/NY.GDP.DEFL.ZS#:~:text=1994%2066,71
MANUAL_MT_GDP_DEFLATOR = {
    1995: 72.86,
    1996: 73.65,
    1997: 74.35,
    1998: 75.26,
    1999: 75.84,
    2000: 68.72,
    2001: 71.76,
    2002: 73.71,
    2003: 75.09,
    2004: 76.68,
    2005: 78.17,
    2006: 79.86,
    2007: 81.68,
    2008: 84.32,
    2009: 86.02,
    2010: 88.74,
    2011: 89.74,
    2012: 91.67,
    2013: 93.75,
    2014: 95.95,
    2015: 100.0,
    2016: 101.76,
    2017: 104.03,
    2018: 106.37,
    2019: 108.84,
    2020: 110.41,
}


def _reshape_series(name: str, code: str, filters: dict, *, annual_mean: bool = False) -> pd.DataFrame:
    df = get_data_df(code, flags=False, filter_pars=filters)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {code} with filters {filters}")

    rename_map = {col: "geo" for col in df.columns if str(col).startswith("geo\\")}
    if rename_map:
        df = df.rename(columns=rename_map)

    time_cols = [col for col in df.columns if TIME_COL_PATTERN.match(str(col))]
    if not time_cols:
        time_cols = [col for col in df.columns if isinstance(col, str) and col[:4].isdigit()]
    if not time_cols:
        raise ValueError(f"Could not detect time columns for {code}")

    id_cols = [col for col in df.columns if col not in time_cols]
    tidy = df.melt(id_vars=id_cols, value_vars=time_cols, var_name="time", value_name=name)
    tidy["time"] = tidy["time"].astype(str)
    tidy = tidy[tidy["time"].str.len() >= 4]
    tidy["time"] = tidy["time"].str[:4].astype(int)
    tidy[name] = pd.to_numeric(tidy[name], errors="coerce")
    tidy = tidy.dropna(subset=[name])
    tidy = tidy[tidy["geo"].isin(COUNTRIES)]
    tidy = tidy[(tidy["time"] >= YEARS.start) & (tidy["time"] <= YEARS.stop - 1)]

    if annual_mean:
        tidy = tidy.groupby(["geo", "time"], as_index=False)[name].mean()
    else:
        tidy = tidy.drop_duplicates(subset=["geo", "time"], keep="last")

    return tidy[["geo", "time", name]]


def _load_legacy_gfcf(path: Path = Path("dataset.xls")) -> pd.DataFrame:
    if not path.exists():
        print(f"Legacy dataset {path} not found; skipping legacy GFCF merge.")
        return pd.DataFrame(columns=["geo", "time", "gfcf_private_legacy_raw"])

    sheet = xlrd.open_workbook(str(path)).sheet_by_index(0)
    header = [str(value).strip() for value in sheet.row_values(0)]
    rows = [sheet.row_values(i) for i in range(1, sheet.nrows)]
    legacy = pd.DataFrame(rows, columns=header).rename(columns={"cou": "geo"})

    if "gfcf_" not in legacy.columns:
        raise ValueError("Column 'gfcf_' missing in legacy dataset")

    legacy = legacy[["geo", "time", "gfcf_"]].rename(columns={"gfcf_": "gfcf_private_legacy_raw"})
    legacy["geo"] = legacy["geo"].astype(str).str.strip()
    legacy["time"] = pd.to_numeric(legacy["time"], errors="coerce").astype("Int64")
    legacy["gfcf_private_legacy_raw"] = pd.to_numeric(legacy["gfcf_private_legacy_raw"], errors="coerce")
    legacy = legacy.dropna(subset=["geo", "time", "gfcf_private_legacy_raw"])

    return legacy


def _load_legacy_population(path: Path = Path("dataset.xls")) -> pd.DataFrame:
    if not path.exists():
        print(f"Legacy dataset {path} not found; skipping legacy population merge.")
        return pd.DataFrame(columns=["geo", "time", "population_legacy_raw"])

    sheet = xlrd.open_workbook(str(path)).sheet_by_index(0)
    header = [str(value).strip() for value in sheet.row_values(0)]
    rows = [sheet.row_values(i) for i in range(1, sheet.nrows)]
    legacy = pd.DataFrame(rows, columns=header).rename(columns={"cou": "geo"})

    if "Population" not in legacy.columns:
        raise ValueError("Column 'Population' missing in legacy dataset")

    legacy = legacy[["geo", "time", "Population"]].rename(columns={"Population": "population_legacy_raw"})
    legacy["geo"] = legacy["geo"].astype(str).str.strip()
    legacy["time"] = pd.to_numeric(legacy["time"], errors="coerce").astype("Int64")
    legacy["population_legacy_raw"] = pd.to_numeric(legacy["population_legacy_raw"], errors="coerce")
    legacy = legacy.dropna(subset=["geo", "time", "population_legacy_raw"])

    return legacy


def _load_public_consumption_total() -> pd.DataFrame:
    base_filters = {
        "freq": "A",
        "sector": "S13",
        "na_item": "P3",
        "unit": "MIO_EUR",
        "time": ANNUAL_TIMES,
    }
    parts = []
    for cofog_code in COFOG_TOTAL_CODES:
        filters = {**base_filters, "cofog99": cofog_code}
        part = _reshape_series("public_consumption_component", "gov_10a_exp", filters)
        parts.append(part)

    combined = pd.concat(parts, ignore_index=True)
    total = (
        combined.groupby(["geo", "time"], as_index=False)["public_consumption_component"].sum()
    )
    return total.rename(columns={"public_consumption_component": "public_consumption"})


def _load_oecd_long_rate(start_year: int = YEARS.start, end_year: int = YEARS.stop - 1) -> pd.DataFrame:
    iso3_list = [ISO2_TO_ISO3[c] for c in COUNTRIES if c in ISO2_TO_ISO3]
    start_period = str(start_year)
    end_period = str(end_year)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv",
    }

    frames = []
    for iso3 in iso3_list:
        url = (
            "https://sdmx.oecd.org/public/rest/data/"
            "OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/"
            f"{iso3}.A.IRLT.PA....."
            f"?startPeriod={start_period}&endPeriod={end_period}"
            "&dimensionAtObservation=AllDimensions"
        )
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            df_raw = pd.read_csv(io.StringIO(resp.text))
        except Exception as exc:  # pragma: no cover
            print(f"OECD long-rate fetch failed for {iso3}: {exc}")
            continue

        if df_raw.empty:
            continue

        df = df_raw[["REF_AREA", "TIME_PERIOD", "OBS_VALUE", "MEASURE", "UNIT_MEASURE", "FREQ"]].copy()
        df = df[(df["MEASURE"] == "IRLT") & (df["UNIT_MEASURE"] == "PA") & (df["FREQ"] == "A")]
        df = df.rename(
            columns={"REF_AREA": "iso3", "TIME_PERIOD": "time", "OBS_VALUE": "long_rate_oecd"}
        )
        df["time"] = pd.to_numeric(df["time"], errors="coerce").astype("Int64")
        iso3_to_iso2 = {v: k for k, v in ISO2_TO_ISO3.items()}
        df["geo"] = df["iso3"].map(iso3_to_iso2)
        df = df.dropna(subset=["geo", "time", "long_rate_oecd"])
        df = df[df["geo"].isin(COUNTRIES)]
        df = df[(df["time"] >= start_year) & (df["time"] <= end_year)]
        frames.append(df[["geo", "time", "long_rate_oecd"]])

    if not frames:
        print("OECD long-rate fetch returned empty dataset")
        return pd.DataFrame(columns=["geo", "time", "long_rate_oecd"])

    return pd.concat(frames, ignore_index=True)


def _load_oecd_outlook_long_rate(path: Path = Path("data") / "Economic_Outlook_116.csv") -> pd.DataFrame:
    if not path.exists():
        print(f"OECD Outlook file {path} not found; skipping")
        return pd.DataFrame(columns=["geo", "time", "long_rate_oecd_outlook"])

    outlook = pd.read_csv(path)

    col_map = {
        "Austria": "AT",
        "Belgium": "BE",
        "Bulgaria": "BG",
        "Croatia": "HR",
        "Czechia": "CZ",
        "Denmark": "DK",
        "Finland": "FI",
        "France": "FR",
        "Germany": "DE",
        "Greece": "EL",
        "Hungary": "HU",
        "Iceland": "IS",
        "Ireland": "IE",
        "Italy": "IT",
        "Latvia": "LV",
        "Lithuania": "LT",
        "Luxembourg": "LU",
        "Netherlands": "NL",
        "Norway": "NO",
        "Poland": "PL",
        "Portugal": "PT",
        "Romania": "RO",
        "Slovak Republic": "SK",
        "Slovenia": "SI",
        "Spain": "ES",
        "Sweden": "SE",
        "Switzerland": "CH",
        "United Kingdom": "UK",
    }

    keep_cols = {name: iso for name, iso in col_map.items() if name in outlook.columns}
    outlook = outlook.rename(columns={"Date": "time", **keep_cols})

    value_cols = list(keep_cols.values())
    if not value_cols:
        print("No OECD Outlook country columns present; skipping")
        return pd.DataFrame(columns=["geo", "time", "long_rate_oecd_outlook"])

    tidy = outlook.melt(
        id_vars=["time"], value_vars=value_cols, var_name="geo", value_name="long_rate_oecd_outlook"
    )
    tidy["time"] = pd.to_numeric(tidy["time"], errors="coerce").astype("Int64")
    tidy["long_rate_oecd_outlook"] = pd.to_numeric(tidy["long_rate_oecd_outlook"], errors="coerce")
    tidy = tidy.dropna(subset=["time"])
    tidy = tidy[tidy["geo"].isin(COUNTRIES)]
    tidy = tidy[(tidy["time"] >= YEARS.start) & (tidy["time"] <= YEARS.stop - 1)]
    tidy = tidy.dropna(subset=["long_rate_oecd_outlook"])

    return tidy[["geo", "time", "long_rate_oecd_outlook"]]


def main() -> None:
    series_frames = []
    for name, meta in TABLES.items():
        series_frames.append(
            _reshape_series(
                name,
                meta["code"],
                meta["filters"],
                annual_mean=meta.get("annual_mean", False),
            )
        )

    series_frames.append(_load_public_consumption_total())
    series_frames.append(_load_oecd_long_rate())
    series_frames.append(_load_oecd_outlook_long_rate())

    panel = reduce(lambda left, right: pd.merge(left, right, on=["geo", "time"], how="outer"), series_frames)

    manual_deflator = pd.DataFrame(
        [
            {"geo": "MT", "time": year, "manual_deflator": value}
            for year, value in MANUAL_MT_GDP_DEFLATOR.items()
        ]
    )
    panel = panel.merge(manual_deflator, on=["geo", "time"], how="left")

    panel["gdp_deflator_2015"] = (panel["gdp_nominal"] / panel["gdp_const_2015"]) * 100
    missing_deflator = panel["gdp_deflator_2015"].isna() & panel["manual_deflator"].notna()
    panel.loc[missing_deflator, "gdp_deflator_2015"] = panel.loc[missing_deflator, "manual_deflator"]
    panel = panel.drop(columns=["manual_deflator"])

    missing_const = panel["gdp_const_2015"].isna() & panel["gdp_deflator_2015"].notna()
    panel.loc[missing_const, "gdp_const_2015"] = (
        panel.loc[missing_const, "gdp_nominal"] / panel.loc[missing_const, "gdp_deflator_2015"] * 100
    )

    derived_real_pc = panel["gdp_const_2015"] * 1_000_000 / panel["population"]
    missing_real_pc = panel["real_gdp_per_capita"].isna() & derived_real_pc.notna()
    panel.loc[missing_real_pc, "real_gdp_per_capita"] = derived_real_pc.loc[missing_real_pc]

    denom = panel["gdp_deflator_2015"].replace({0: pd.NA})
    panel["env_consumption_const_2015"] = panel["env_consumption"] / denom * 100
    panel["env_investment_const_2015"] = panel["env_investment"] / denom * 100
    panel["env_total_const_2015"] = panel["env_consumption_const_2015"] + panel["env_investment_const_2015"]
    panel["env_total"] = panel["env_investment"] + panel["env_consumption"]
    panel["public_consumption_const_2015"] = panel["public_consumption"] / denom * 100

    panel["gfcf_private"] = panel["gfcf_total"] - panel["gfcf_public"]
    panel["gfcf_total_const_2015"] = panel["gfcf_total"] / denom * 100
    panel["gfcf_public_const_2015"] = panel["gfcf_public"] / denom * 100
    panel["gfcf_private_const_2015"] = panel["gfcf_private"] / denom * 100

    panel["gdp_nominal_bln"] = panel["gdp_nominal"] / 1000.0
    panel["value_added_per_1000_workers"] = panel["value_added"] / panel["workers_thousands"].replace({0: pd.NA})

    panel["long_rate_filled"] = panel["long_rate"]
    panel["long_rate_source"] = pd.NA
    eurostat_mask = panel["long_rate"].notna()
    panel.loc[eurostat_mask, "long_rate_source"] = "eurostat"

    missing_long = panel["long_rate_filled"].isna() & panel["long_rate_oecd"].notna()
    panel.loc[missing_long, "long_rate_filled"] = panel.loc[missing_long, "long_rate_oecd"]
    panel.loc[missing_long, "long_rate_source"] = "oecd_mei_fin"

    missing_long_outlook = panel["long_rate_filled"].isna() & panel["long_rate_oecd_outlook"].notna()
    panel.loc[missing_long_outlook, "long_rate_filled"] = panel.loc[missing_long_outlook, "long_rate_oecd_outlook"]
    panel.loc[missing_long_outlook, "long_rate_source"] = "oecd_outlook_116"

    missing_long_ecb = panel["long_rate_filled"].isna() & panel["long_rate_ecb"].notna()
    panel.loc[missing_long_ecb, "long_rate_filled"] = panel.loc[missing_long_ecb, "long_rate_ecb"]
    panel.loc[missing_long_ecb, "long_rate_source"] = "ecb_mcby"

    legacy_gfcf = _load_legacy_gfcf()
    if legacy_gfcf.empty:
        panel["gfcf_private_legacy_raw"] = pd.NA
    else:
        panel = panel.merge(legacy_gfcf, on=["geo", "time"], how="left")

    legacy_population = _load_legacy_population()
    if legacy_population.empty:
        panel["population_legacy_raw"] = pd.NA
    else:
        panel = panel.merge(legacy_population, on=["geo", "time"], how="left")

    panel = panel.sort_values(["geo", "time"]).reset_index(drop=True)

    out_path = Path("data") / "eurostat_green_panel.csv"
    out_path.parent.mkdir(exist_ok=True)
    try:
        panel.to_csv(out_path, index=False)
        print(f"Saved {len(panel)} rows to {out_path}")
    except PermissionError:
        fallback = Path("data") / "eurostat_green_panel_unlocked.csv"
        panel.to_csv(fallback, index=False)
        print(
            f"Permission denied on {out_path}; wrote {len(panel)} rows to {fallback} instead."
        )


if __name__ == "__main__":
    main()
