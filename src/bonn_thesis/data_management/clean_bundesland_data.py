"""Functions to clean German city/state reference data."""

import pandas as pd

from bonn_thesis.config import BUNDESLAND_MAP, CITIES_ENGLISH
from bonn_thesis.data_management.location_normalize_string import (
    normalize_location_string,
)


def clean_bundesland_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize German city and state reference data.

    Args:
        raw_df (pd.DataFrame): Raw data from Excel file with columns:
            - Amtlicher Regionalschlüssel
            - Stadt
            - Postleitzahl
            - Bevölkerung auf Grundlage des ZENSUS 2022 ²⁾ insgesamt

    Returns:
        pd.DataFrame: Cleaned dataframe with normalized city and state names
            in German and English. Duplicated cities are removed, keeping the one
            with the highest population.
    """
    clean_df = raw_df[
        [
            "Amtlicher Regionalschlüssel",
            "Stadt",
            "Postleitzahl ",
            "Bevölkerung auf Grundlage des ZENSUS 2022 ²⁾ insgesamt",
        ]
    ].rename(
        columns={
            "Amtlicher Regionalschlüssel": "reg_code",
            "Stadt": "city",
            "Postleitzahl ": "plz",
            "Bevölkerung auf Grundlage des ZENSUS 2022 ²⁾ insgesamt": "population",
        }
    )

    clean_df["city"] = clean_df["city"].astype(str)
    clean_df["plz"] = clean_df["plz"].astype(str).str.zfill(5)
    clean_df["reg_code"] = clean_df["reg_code"].astype(str).str.zfill(12)
    clean_df["population"] = clean_df["population"].astype(int)

    clean_df["bland_code"] = clean_df["reg_code"].str[:2]

    clean_df["city"] = clean_df["city"].apply(_clean_city_name)

    clean_df = clean_df[["bland_code", "reg_code", "plz", "city", "population"]]

    clean_df = (
        clean_df.sort_values("population", ascending=False)
        .drop_duplicates(subset=["city"], keep="first")
        .sort_values("city")
        .reset_index(drop=True)
    )

    english_df = pd.DataFrame(CITIES_ENGLISH)
    clean_df = pd.concat([clean_df, english_df], ignore_index=True)

    clean_df["state_de"] = clean_df["bland_code"].map(lambda x: BUNDESLAND_MAP[x][0])
    clean_df["state_en"] = clean_df["bland_code"].map(lambda x: BUNDESLAND_MAP[x][1])

    clean_df["city_normalized"] = clean_df["city"].apply(normalize_location_string)
    clean_df["state_de_normalized"] = clean_df["state_de"].apply(
        normalize_location_string
    )
    clean_df["state_en_normalized"] = clean_df["state_en"].apply(
        normalize_location_string
    )
    return clean_df


def _clean_city_name(city: str) -> str:
    """Remove text after comma and parentheses from city name,
    with exception for Frankfurt    .
    """
    if "," in city:
        city = city.split(",")[0]
    if "(" in city and "Frankfurt (Oder)" not in city:  # codespell:ignore
        city = city.split(" (")[0]
    if "/" in city:
        city = city.split(" /")[0]
        city = city.split("/")[0]
    return city.strip()
