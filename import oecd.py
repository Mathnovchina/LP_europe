import io
import requests
import pandas as pd

def fetch_oecd_irlt(countries_iso3, start_year=1995, end_year=2022, freq="A"):
    """
    Fetch long-term interest rates (IRLT, 10y gov bond yields, % p.a.)
    from OECD MEI_FIN via the new SDMX API.

    countries_iso3: list like ['AUT','BEL','CHE',...]
    freq: 'A' (annual) or 'M' (monthly)
    """
    database = "@DF_FINMARK,4.0"      # MEI_FIN financial market dataset
    indicator = "IRLT"                # long-term interest rates
    unit = "PA"                       # per annum

    country_code = "+".join(countries_iso3)

    if freq == "A":
        start_period = str(start_year)
        end_period = str(end_year)
    elif freq == "M":
        start_period = f"{start_year}-01"
        end_period = f"{end_year}-12"
    else:
        raise ValueError("freq must be 'A' or 'M'")

    query_text = (
        f"{database}/"
        f"{country_code}.{freq}.{indicator}.{unit}....."
        f"?startPeriod={start_period}"
        f"&endPeriod={end_period}"
        f"&dimensionAtObservation=AllDimensions"
    )

    url = f"https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES{query_text}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/vnd.sdmx.data+csv; charset=utf-8",
    }

    r = requests.get(url, headers=headers)
    r.raise_for_status()

    df_raw = pd.read_csv(io.StringIO(r.text))

    # Keep the essentials
    df = df_raw[["REF_AREA", "FREQ", "MEASURE", "UNIT_MEASURE",
                 "TIME_PERIOD", "OBS_VALUE"]].copy()

    # (Should already hold, but just to be safe)
    df = df[
        (df["MEASURE"] == "IRLT") &
        (df["UNIT_MEASURE"] == "PA") &
        (df["FREQ"] == freq)
    ]

    if freq == "A":
        df.rename(
            columns={
                "REF_AREA": "iso3",
                "TIME_PERIOD": "year",
                "OBS_VALUE": "long_rate",
            },
            inplace=True,
        )
        df["year"] = df["year"].astype(int)
    else:  # monthly
        df.rename(
            columns={
                "REF_AREA": "iso3",
                "TIME_PERIOD": "month",
                "OBS_VALUE": "long_rate",
            },
            inplace=True,
        )
        df["year"] = df["month"].str.slice(0, 4).astype(int)

    return df


# European countries you care about (OECD / partners)
EURO_ISO3 = [
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
    "DEU", "GRC", "HUN", "IRL", "ISL", "ITA", "LVA", "LTU", "LUX", "MLT",
    "NLD", "NOR", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE", "CHE",
    "GBR",
]

oecd_long = fetch_oecd_irlt(EURO_ISO3, start_year=1995, end_year=2022, freq="A")

print(oecd_long.head())
