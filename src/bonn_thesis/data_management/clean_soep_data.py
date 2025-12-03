"""Clean and transform SOEP survey data."""

import pandas as pd

from bonn_thesis.config import MAX_DATA_YEAR, MIN_DATA_YEAR


def clean_soep_data(
    pgen: pd.DataFrame,
    ppathl: pd.DataFrame,
    hbrutto: pd.DataFrame,
    isco: pd.DataFrame,
) -> pd.DataFrame:
    """Clean and merge SOEP data with derived variables.

    Merges person-level, household, and education data, then creates cleaned
    variables for occupation codes (ISCO, KLDB), education levels (ISCED),
    demographics, and location.

    Args:
        pgen: Person-level generated data with occupation and income
        ppathl: Person-level path data with demographics
        hbrutto: Household-level data with location
        isco: ISCO occupation classification reference data

    Returns:
        Cleaned DataFrame with inserted variables
    """
    isco_lookups = _build_isco_lookups(isco)

    partial_merge = pgen.merge(ppathl, on=["hid", "pid", "syear"])
    merge = partial_merge.merge(hbrutto, on=["hid", "syear"])

    merge = _extract_isco_codes(merge, isco_lookups)
    merge = _extract_kldb_codes(merge)
    merge = _extract_education_codes(merge)
    merge = _transform_demographics(merge)
    merge = _convert_numeric_columns(merge)

    filtered_df = _filter_valid_records(merge)

    return filtered_df


def _build_isco_lookups(isco: pd.DataFrame) -> dict:
    """Build ISCO code lookup dictionaries for all digit levels."""
    lookups = {}

    for level in range(1, 5):
        isco_level = isco[isco["Level"] == level].copy()
        isco_level["ISCO 08 Code"] = (
            isco_level["ISCO 08 Code"].astype(str).str.zfill(level)
        )
        lookups[level] = isco_level.set_index("ISCO 08 Code")["Title EN"].to_dict()

    return lookups


def _extract_isco_codes(merge: pd.DataFrame, isco_lookups: dict) -> pd.DataFrame:
    """Extract and map ISCO occupation codes."""
    merge["isco_code"] = merge["pgisco08"].astype(str).str.extract(r"\[(\d+)\]")[0]
    merge["isco_code"] = merge["isco_code"].str.zfill(4)

    merge["isco_3_digit"] = merge["isco_code"].str[:3]
    merge["isco_2_digit"] = merge["isco_code"].str[:2]
    merge["isco_1_digit"] = merge["isco_code"].str[:1]

    merge["isco_1_name"] = merge["isco_1_digit"].map(isco_lookups[1])
    merge["isco_2_name"] = merge["isco_2_digit"].map(isco_lookups[2])
    merge["isco_3_name"] = merge["isco_3_digit"].map(isco_lookups[3])
    merge["isco_4_name"] = merge["isco_code"].map(isco_lookups[4])

    return merge


def _extract_kldb_codes(merge: pd.DataFrame) -> pd.DataFrame:
    """Extract KLDB occupation codes (German classification)."""
    merge["kldb_code"] = merge["pgkldb2010"].astype(str).str.extract(r"\[(\d+)\]")[0]
    merge["kldb_code"] = merge["kldb_code"].str.zfill(5)

    merge["kldb_4_digit"] = merge["kldb_code"].str[:4]
    merge["kldb_3_digit"] = merge["kldb_code"].str[:3]
    merge["kldb_2_digit"] = merge["kldb_code"].str[:2]
    merge["kldb_1_digit"] = merge["kldb_code"].str[:1]
    merge["kldb_skill"] = merge["kldb_code"].str[4:5]

    return merge


def _extract_education_codes(merge: pd.DataFrame) -> pd.DataFrame:
    """Extract and group education levels (ISCED)."""
    merge["isced_code"] = pd.to_numeric(
        merge["pgisced11"].astype(str).str.extract(r"\[(\d+)\]")[0],
        errors="coerce",
    ).astype("Int64")
    merge["education"] = (
        merge["pgisced11"].astype(str).str.extract(r"\[\d+\]\s*(.+)")[0]
    )

    merge["education_grouped"] = merge["isced_code"].map(
        {
            1: "Primary education",
            2: "Secondary education",
            3: "Secondary education",
            4: "Post-secondary non-tertiary or Short-cycle tertiary education",
            5: "Post-secondary non-tertiary or Short-cycle tertiary education",
            6: "Bachelor degree",
            7: "Master or Doctoral degree",
            8: "Master or Doctoral degree",
        }
    )

    return merge


def _transform_demographics(merge: pd.DataFrame) -> pd.DataFrame:
    """Transform demographic variables to English."""
    merge["sex_en"] = (
        merge["sex"].astype(str).str.replace(r"\[1\] maennlich", "male", regex=True)
    )
    merge["sex_en"] = merge["sex_en"].str.replace(
        r"\[2\] weiblich", "female", regex=True
    )

    merge["bland_code"] = pd.to_numeric(
        merge["bula_h"].astype(str).str.extract(r"\[(\d+)\]")[0],
        errors="coerce",
    ).astype("Int64")

    return merge


def _convert_numeric_columns(merge: pd.DataFrame) -> pd.DataFrame:
    """Convert income and experience columns to numeric."""
    merge["pglabgro"] = pd.to_numeric(merge["pglabgro"], errors="coerce")
    merge["pglabnet"] = pd.to_numeric(merge["pglabnet"], errors="coerce")
    merge["pgexpft"] = pd.to_numeric(merge["pgexpft"], errors="coerce")

    return merge


def _filter_valid_records(merge: pd.DataFrame) -> pd.DataFrame:
    """Filter for valid records within time period and with valid responses."""
    filtered_df = merge[
        (merge["syear"] >= MIN_DATA_YEAR)
        & (merge["syear"] <= MAX_DATA_YEAR)
        & (merge["pgisco08"] != "[-2] trifft nicht zu")
        & (merge["pgisced11"] != "[-1] keine Angabe")
        & (merge["pgisced11"] != "[-2] trifft nicht zu")
        & (merge["sex"] != "[-3] nicht valide")  # codespell: ignore
        & (merge["pgexpft"] != "[-1] keine Angabe")
        & (
            merge["pgisco08"]
            != "[-8] Frage in diesem Jahr nicht Teil des Frageprograms"
        )
        & (merge["pgisco08"] != "[-1] keine Angabe")
        & (merge["pgisced11"] != "[0] in school")
    ].copy()

    filtered_df = filtered_df[
        filtered_df["pglabgro"].notna()
        | filtered_df["pglabnet"].notna()
        | filtered_df["pgexpft"].notna()
    ]

    return filtered_df
