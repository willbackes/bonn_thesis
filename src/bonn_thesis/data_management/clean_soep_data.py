"""Clean and transform SOEP survey data."""

import pandas as pd

from bonn_thesis.config import MAX_DATA_YEAR, MIN_DATA_YEAR


def clean_soep_data(
    pgen: pd.DataFrame,
    ppathl: pd.DataFrame,
    hbrutto: pd.DataFrame,
    isco: pd.DataFrame,
    bundesland_data: pd.DataFrame,
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
        bundesland_data: Federal state reference data with state names

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
    merge = _add_federal_state_names(merge, bundesland_data)

    filtered_df = _filter_valid_records(merge)

    filtered_df = filtered_df[filtered_df["pgemplst"] == "[1] Voll erwerbstätig"].copy()

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
    result = merge.copy()
    result["isco_code"] = result["pgisco08"].astype(str).str.extract(r"\[(\d+)\]")[0]
    result["isco_code"] = result["isco_code"].str.zfill(4)

    result["isco_3_digit"] = result["isco_code"].str[:3]
    result["isco_2_digit"] = result["isco_code"].str[:2]
    result["isco_1_digit"] = result["isco_code"].str[:1]

    result["isco_1_name"] = result["isco_1_digit"].map(isco_lookups[1])
    result["isco_2_name"] = result["isco_2_digit"].map(isco_lookups[2])
    result["isco_3_name"] = result["isco_3_digit"].map(isco_lookups[3])
    result["isco_4_name"] = result["isco_code"].map(isco_lookups[4])

    return result


def _extract_kldb_codes(merge: pd.DataFrame) -> pd.DataFrame:
    """Extract KLDB occupation codes (German classification)."""
    result = merge.copy()
    result["kldb_code"] = result["pgkldb2010"].astype(str).str.extract(r"\[(\d+)\]")[0]
    result["kldb_code"] = result["kldb_code"].str.zfill(5)

    result["kldb_4_digit"] = result["kldb_code"].str[:4]
    result["kldb_3_digit"] = result["kldb_code"].str[:3]
    result["kldb_2_digit"] = result["kldb_code"].str[:2]
    result["kldb_1_digit"] = result["kldb_code"].str[:1]
    result["kldb_skill"] = result["kldb_code"].str[4:5]

    return result


def _extract_education_codes(merge: pd.DataFrame) -> pd.DataFrame:
    """Extract and group education levels (ISCED)."""
    result = merge.copy()
    result["isced_code"] = pd.to_numeric(
        result["pgisced11"].astype(str).str.extract(r"\[(\d+)\]")[0],
        errors="coerce",
    ).astype("Int64")
    result["education"] = (
        result["pgisced11"].astype(str).str.extract(r"\[\d+\]\s*(.+)")[0]
    )

    result["education_grouped"] = result["isced_code"].map(
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

    return result


def _transform_demographics(merge: pd.DataFrame) -> pd.DataFrame:
    """Transform demographic variables to English."""
    result = merge.copy()
    result["sex_en"] = (
        result["sex"].astype(str).str.replace(r"\[1\] maennlich", "male", regex=True)
    )
    result["sex_en"] = result["sex_en"].str.replace(
        r"\[2\] weiblich", "female", regex=True
    )

    result["bland_code"] = pd.to_numeric(
        result["bula_h"].astype(str).str.extract(r"\[(\d+)\]")[0],
        errors="coerce",
    ).astype("Int64")

    return result


def _add_federal_state_names(
    merge: pd.DataFrame, bundesland_data: pd.DataFrame
) -> pd.DataFrame:
    """Add federal state names based on federal state codes."""
    result = merge.copy()
    bundesland_lookup = bundesland_data.copy()
    bundesland_lookup["bland_code"] = bundesland_lookup["bland_code"].astype(int)
    bundesland_unique = bundesland_lookup.drop_duplicates("bland_code").set_index(
        "bland_code"
    )

    result["state_en"] = result["bland_code"].map(bundesland_unique["state_en"])
    result["state_de"] = result["bland_code"].map(bundesland_unique["state_de"])

    return result


def _convert_numeric_columns(merge: pd.DataFrame) -> pd.DataFrame:
    """Convert income and experience columns to numeric."""
    result = merge.copy()
    result["pglabgro"] = pd.to_numeric(result["pglabgro"], errors="coerce")
    result["pglabnet"] = pd.to_numeric(result["pglabnet"], errors="coerce")
    result["pgexpft"] = pd.to_numeric(result["pgexpft"], errors="coerce")

    return result


def _filter_valid_records(merge: pd.DataFrame) -> pd.DataFrame:
    """Filter for valid records within time period and with valid responses."""
    result = merge.copy()
    filtered_df = result[
        (result["syear"] >= MIN_DATA_YEAR)
        & (result["syear"] <= MAX_DATA_YEAR)
        & (result["pgisco08"] != "[-2] trifft nicht zu")
        & (result["pgisced11"] != "[-1] keine Angabe")
        & (result["pgisced11"] != "[-2] trifft nicht zu")
        & (result["sex"] != "[-3] nicht valide")  # codespell: ignore
        & (result["pgexpft"] != "[-1] keine Angabe")
        & (
            result["pgisco08"]
            != "[-8] Frage in diesem Jahr nicht Teil des Frageprograms"
        )
        & (result["pgisco08"] != "[-1] keine Angabe")
        & (result["pgisced11"] != "[0] in school")
    ].copy()

    filtered_df = filtered_df[
        filtered_df["pglabgro"].notna()
        | filtered_df["pglabnet"].notna()
        | filtered_df["pgexpft"].notna()
    ]

    return filtered_df
