"""Functions for merging experience, education, and ISCO classification data."""

import json
import re
from pathlib import Path

import pandas as pd

from bonn_thesis.config import EDUCATION_HIERARCHY


def parse_isco_jsonl_simple(file_path):
    """Parse ISCO JSONL file and extract exp_id and codes.

    Args:
        file_path: Path to JSONL file with ISCO classification results

    Returns:
        DataFrame with columns: exp_id, isco_3_digit
    """
    results = []

    with Path(file_path).open(encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            custom_id = data.get("custom_id", "")
            match = re.match(r"isco_classification_(\d+)_(\d+)", custom_id)

            if not match:
                continue

            exp_id = int(match.group(2))

            # Extract response content
            response_content = None
            if "response" in data and "body" in data["response"]:
                body = data["response"]["body"]
                if "choices" in body and len(body["choices"]) > 0:
                    message = body["choices"][0].get("message", {})
                    response_content = message.get("content", "")

            # Extract ISCO code
            if response_content and isinstance(response_content, str):
                code_match = re.search(r"\b([0-9]{3})\b", response_content)
                isco_code = (
                    code_match.group(1)
                    if code_match and code_match.group(1) != "000"
                    else None
                )
            else:
                isco_code = None

            results.append(
                {
                    "exp_id": exp_id,
                    "isco_3_digit": isco_code,
                }
            )

    return pd.DataFrame(results)


def merge_isco_codes(sample_df, isco_df):
    """Merge ISCO classification codes with sample selection data.

    Args:
        sample_df: DataFrame with experience data
        isco_df: DataFrame with ISCO codes (exp_id, isco_3_digit)

    Returns:
        DataFrame with ISCO codes merged
    """
    merged_df = sample_df.merge(
        isco_df[["exp_id", "isco_3_digit"]], on="exp_id", how="left"
    )

    return merged_df


def filter_years(df, min_year=2013, max_year=2019):
    """Filter dataframe to keep only observations within specified year range.

    Args:
        df: DataFrame with 'syear' column
        min_year: Minimum year to keep (inclusive)
        max_year: Maximum year to keep (inclusive)

    Returns:
        Filtered DataFrame
    """
    filtered_df = df[(df["syear"] >= min_year) & (df["syear"] <= max_year)].copy()

    return filtered_df


def filter_missing_isco(df):
    """Remove rows where isco_3_digit is missing.

    Args:
        df: DataFrame with isco_3_digit column

    Returns:
        Filtered DataFrame with only rows that have valid ISCO codes
    """
    filtered_df = df[df["isco_3_digit"].notna()].copy()

    return filtered_df


def expand_to_yearly_observations(df):
    """Expand experience data to yearly observations.

    For each job experience, create one row per year the job spans,
    with observations dated December 31st of each year.

    Args:
        df: DataFrame with job experience data including:
            - exp_start_date: Job start date
            - exp_end_date: Job end date
            - experience_at_start_ft: Experience at job start (in years)

    Returns:
        DataFrame with yearly observations including:
            - syear: Year of observation (float)
            - pgexpft: Experience at the observation date (float)
            - All other original columns
    """
    expanded_rows = []

    for _idx, row in df.iterrows():
        start_date = pd.Timestamp(row["exp_start_date"])
        end_date = pd.Timestamp(row["exp_end_date"])

        # Get the years this job spans
        start_year = start_date.year
        end_year = end_date.year

        # For each year in the span
        for year in range(start_year, end_year + 1):
            # Create observation date (December 31st of the year)
            obs_date = pd.Timestamp(f"{year}-12-31")

            # Only include this year if:
            # 1. The job started before or on Dec 31 of this year
            # 2. The job ended after or on Dec 31 of this year
            if start_date <= obs_date <= end_date:
                # Calculate experience at observation date
                days_from_start = (obs_date - start_date).days
                years_from_start = days_from_start / 365.25

                pgexpft = row["experience_at_start_ft"] + years_from_start

                # Create new row with all original data
                new_row = row.copy()
                new_row["syear"] = float(year)
                new_row["pgexpft"] = float(pgexpft)

                expanded_rows.append(new_row)

    # Create DataFrame from expanded rows
    df_expanded = pd.DataFrame(expanded_rows)

    return df_expanded


def merge_education_yearly(yearly_df, education_df):
    """Merge education data with yearly observations.

    For each yearly observation, assign the highest education level obtained
    by that year (where ed_end_date <= syear, December 31st).

    If no education is found, assume "Secondary education".

    Args:
        yearly_df: DataFrame with yearly observations (includes syear)
        education_df: DataFrame with education data
            (includes ed_end_date, education_grouped)

    Returns:
        DataFrame with education columns added:
            ed_id, ed_start_date, ed_end_date, education_grouped
    """
    # Create a copy to avoid modifying the original
    result_df = yearly_df.copy()

    # Initialize education columns
    result_df["ed_id"] = pd.Series(dtype="Int64")
    result_df["ed_start_date"] = pd.Series(dtype="datetime64[ns]")
    result_df["ed_end_date"] = pd.Series(dtype="datetime64[ns]")
    result_df["education_grouped"] = pd.Series(dtype="object")

    # Add rank column to education data
    edu_with_rank = education_df.copy()
    edu_with_rank["education_rank"] = edu_with_rank["education_grouped"].map(
        EDUCATION_HIERARCHY
    )

    # Filter out education records with NaN values
    edu_with_rank = edu_with_rank[
        edu_with_rank["education_grouped"].notna()
        & edu_with_rank["education_rank"].notna()
    ].copy()

    # Create observation date (December 31st of each year)
    result_df["obs_date"] = pd.to_datetime(
        result_df["syear"].astype(int).astype(str) + "-12-31"
    )

    # Process each person
    for prof_id, group in result_df.groupby("prof_id"):
        # Get education records for this person
        person_edu = edu_with_rank[edu_with_rank["prof_id"] == prof_id].copy()

        if len(person_edu) == 0:
            # No education data - assign "Secondary education"
            result_df.loc[group.index, "education_grouped"] = "Secondary education"
            continue

        # For each year observation
        for idx, row in group.iterrows():
            obs_date = row["obs_date"]

            # Find all education completed by this year
            completed_edu = person_edu[person_edu["ed_end_date"] <= obs_date]

            if len(completed_edu) == 0:
                # No education completed yet
                result_df.loc[idx, "education_grouped"] = "Secondary education"
            else:
                # Take the highest education level
                highest_edu = completed_edu.loc[
                    completed_edu["education_rank"].idxmax()
                ]

                result_df.loc[idx, "ed_id"] = highest_edu["ed_id"]
                result_df.loc[idx, "ed_start_date"] = highest_edu["ed_start_date"]
                result_df.loc[idx, "ed_end_date"] = highest_edu["ed_end_date"]
                result_df.loc[idx, "education_grouped"] = highest_edu[
                    "education_grouped"
                ]

    # Drop temporary column
    result_df = result_df.drop(columns=["obs_date"])

    return result_df


def add_isco_digit_levels(df):
    """Extract ISCO digit levels from isco_3_digit code.

    Args:
        df: DataFrame with isco_3_digit column

    Returns:
        DataFrame with isco_1_digit and isco_2_digit columns added
    """
    result_df = df.copy()

    # Extract digit levels
    result_df["isco_2_digit"] = result_df["isco_3_digit"].astype(str).str[:2]
    result_df["isco_1_digit"] = result_df["isco_3_digit"].astype(str).str[:1]

    # Replace 'na' with None for consistency
    result_df.loc[result_df["isco_3_digit"].isna(), "isco_2_digit"] = None
    result_df.loc[result_df["isco_3_digit"].isna(), "isco_1_digit"] = None

    return result_df


def build_isco_lookups(isco_df):
    """Build ISCO code lookup dictionaries for all digit levels.

    Args:
        isco_df: DataFrame with ISCO reference data
            (columns: Level, ISCO 08 Code, Title EN)

    Returns:
        Dictionary with lookups for levels 1, 2, and 3
    """
    lookups = {}

    for level in [1, 2, 3]:
        isco_level = isco_df[isco_df["Level"] == level].copy()
        isco_level["ISCO 08 Code"] = (
            isco_level["ISCO 08 Code"].astype(str).str.zfill(level)
        )
        lookups[level] = isco_level.set_index("ISCO 08 Code")["Title EN"].to_dict()

    return lookups


def add_isco_names(df, isco_lookups):
    """Add ISCO occupation names based on code lookups.

    Args:
        df: DataFrame with isco_1_digit, isco_2_digit, isco_3_digit columns
        isco_lookups: Dictionary with ISCO code to name mappings

    Returns:
        DataFrame with isco_1_name, isco_2_name, isco_3_name columns added
    """
    result_df = df.copy()

    result_df["isco_1_name"] = result_df["isco_1_digit"].map(isco_lookups[1])
    result_df["isco_2_name"] = result_df["isco_2_digit"].map(isco_lookups[2])
    result_df["isco_3_name"] = result_df["isco_3_digit"].map(isco_lookups[3])

    return result_df


def transform_gender_to_english(df):
    """Transform gender column to English (sex_en).

    Args:
        df: DataFrame with gender column

    Returns:
        DataFrame with sex_en column added
    """
    result_df = df.copy()

    # Map gender values to English
    gender_map = {
        "M": "male",
        "F": "female",
    }

    result_df["sex_en"] = result_df["gender"].map(gender_map)

    # If gender is already in correct format, use it directly
    if result_df["sex_en"].isna().all() and result_df["gender"].notna().any():
        result_df["sex_en"] = result_df["gender"]

    return result_df


def add_state_names(df, bundesland_df):
    """Add federal state names (German and English) based on bland_code.

    Args:
        df: DataFrame with bland_code column
        bundesland_df: DataFrame with bundesland reference data
            (columns: bland_code, state_de, state_en)

    Returns:
        DataFrame with state_de and state_en columns added
    """
    result_df = df.copy()

    # Prepare bundesland lookup
    bundesland_lookup = bundesland_df.copy()
    bundesland_lookup["bland_code"] = bundesland_lookup["bland_code"].astype(int)
    bundesland_unique = bundesland_lookup.drop_duplicates("bland_code").set_index(
        "bland_code"
    )

    # Map state names
    result_df["state_en"] = result_df["bland_code"].map(bundesland_unique["state_en"])
    result_df["state_de"] = result_df["bland_code"].map(bundesland_unique["state_de"])

    return result_df


def merge_exp_ed_pipeline(
    isco_file,
    sample_file,
    education_file,
    isco_reference_df,
    bundesland_reference_df,
):
    """Run complete merge pipeline for experience, education, and ISCO data.

    Args:
        isco_file: Path to ISCO classification JSONL file
        sample_file: Path to sample selection parquet file
        education_file: Path to education data parquet file
        isco_reference_df: DataFrame with ISCO reference data
        bundesland_reference_df: DataFrame with bundesland reference data

    Returns:
        DataFrame with merged and enriched data
    """
    # Step 1: Parse ISCO codes
    isco_df = parse_isco_jsonl_simple(isco_file)

    # Step 2: Load sample selection data
    sample_df = pd.read_parquet(sample_file, engine="fastparquet")

    # Step 3: Load education data
    education_df = pd.read_parquet(education_file, engine="fastparquet")

    # Step 4: Merge ISCO codes with sample data
    df_with_isco = merge_isco_codes(sample_df, isco_df)

    # Step 5: Filter out rows without ISCO codes
    df_with_isco_filtered = filter_missing_isco(df_with_isco)

    # Step 6: Expand to yearly observations
    df_yearly = expand_to_yearly_observations(df_with_isco_filtered)

    # Step 7: Filter to 2013-2019
    df_filtered = filter_years(df_yearly, min_year=2013, max_year=2019)

    # Step 8: Merge education data
    df_with_education = merge_education_yearly(df_filtered, education_df)

    # Step 9: Add ISCO digit levels
    df_with_digits = add_isco_digit_levels(df_with_education)

    # Step 10: Add ISCO occupation names
    isco_lookups = build_isco_lookups(isco_reference_df)
    df_with_names = add_isco_names(df_with_digits, isco_lookups)

    # Step 11: Transform gender to English
    df_with_sex = transform_gender_to_english(df_with_names)

    # Step 12: Add federal state names
    df_complete = add_state_names(df_with_sex, bundesland_reference_df)

    return df_complete
