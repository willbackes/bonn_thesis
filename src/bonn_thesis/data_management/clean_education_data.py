"""Functions to clean LinkedIn education data."""

import pandas as pd

from bonn_thesis.config import (
    EDUCATION_MAPPING,
    MAX_SAFE_DATE,
    MIN_SAFE_DATE,
)
from bonn_thesis.data_management.education_identify_degree import clean_degree_type


def clean_education_data(
    education_df: pd.DataFrame, mapping_dict: dict
) -> pd.DataFrame:
    """Clean LinkedIn education data.

    Args:
        education_df: Raw education DataFrame with columns:
            - ed_id, prof_id, degree_type, start_date, end_date, case_degree_label
        mapping_dict: Dictionary mapping degree_type_cleaned to education_level

    Returns:
        Cleaned DataFrame with columns:
            - ed_id, prof_id, start_date, end_date, education_grouped
    """
    clean_df = pd.DataFrame()

    education_df["start_date"] = pd.to_datetime(
        education_df["start_date"], errors="coerce"
    )
    education_df["end_date"] = pd.to_datetime(education_df["end_date"], errors="coerce")

    invalid_start = (education_df["start_date"] < MIN_SAFE_DATE) | (
        education_df["start_date"] > MAX_SAFE_DATE
    )
    invalid_end = (education_df["end_date"] < MIN_SAFE_DATE) | (
        education_df["end_date"] > MAX_SAFE_DATE
    )

    education_df.loc[invalid_start, "start_date"] = pd.NaT
    education_df.loc[invalid_end, "end_date"] = pd.NaT

    valid_mask = education_df["start_date"].notna() & education_df["end_date"].notna()
    valid_records = education_df[valid_mask].copy()

    clean_df["ed_id"] = valid_records["ed_id"]
    clean_df["prof_id"] = valid_records["prof_id"]
    clean_df["ed_start_date"] = valid_records["start_date"]
    clean_df["ed_end_date"] = valid_records["end_date"]
    clean_df["degree_type"] = valid_records["degree_type"]
    clean_df["case_degree_label"] = valid_records["case_degree_label"]

    clean_df["degree_type_cleaned"] = clean_degree_type(clean_df["degree_type"])

    clean_df["education_grouped"] = clean_df["case_degree_label"].map(EDUCATION_MAPPING)

    clean_df["education_grouped"] = clean_df["education_grouped"].fillna(
        clean_df["degree_type_cleaned"].map(mapping_dict)
    )

    profiles_with_mapping = clean_df[clean_df["education_grouped"].notna()][
        "prof_id"
    ].unique()
    clean_df = clean_df.drop(
        clean_df[~clean_df["prof_id"].isin(profiles_with_mapping)].index
    )

    columns_to_keep = [
        "ed_id",
        "prof_id",
        "ed_start_date",
        "ed_end_date",
        "education_grouped",
    ]
    columns_to_drop = [col for col in clean_df.columns if col not in columns_to_keep]
    clean_df = clean_df.drop(columns=columns_to_drop)

    return clean_df
