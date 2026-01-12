"""Sample selection functions for filtering experience data."""

import pandas as pd

from bonn_thesis.config import (
    EDUCATION_HIERARCHY,
    EXCLUDED_COMP_ID,
    EXCLUDED_MIN_SIZE,
)


def run_sample_selection(experience_data, education_data):
    """Run complete sample selection pipeline.

    Args:
        experience_data: DataFrame with raw experience data
        education_data: DataFrame with education data

    Returns:
        Tuple of (filtered_data, tracking_log)
    """
    tracking_log = []

    # Compute experience at start
    experience_data = compute_experience_at_start_ft(experience_data)

    # Apply filters
    filtered_data = apply_sample_selection_filters(experience_data, tracking_log)

    # Select columns
    filtered_data = select_final_columns(filtered_data)

    # Clean company names
    filtered_data = clean_company_names(filtered_data)

    # Prepare education data
    educ_prepared = prepare_education_for_merge(education_data)

    # Merge education
    merged_data = merge_education_by_groups(filtered_data, educ_prepared)

    # Apply education filter
    final_data = apply_education_filter(merged_data, tracking_log)

    # Drop education columns
    columns_to_drop = ["ed_id", "ed_start_date", "ed_end_date", "education_level"]
    final_data = final_data.drop(
        columns=[col for col in columns_to_drop if col in final_data.columns]
    )

    return final_data, tracking_log


def calculate_non_overlapping_months(date_ranges):
    """Calculate total months covered by date ranges, counting overlaps only once.

    Args:
        date_ranges: Array of shape (n, 2) with start and end dates

    Returns:
        Total non-overlapping months
    """
    if len(date_ranges) == 0:
        return 0.0

    # Convert to list and sort by start date
    intervals = sorted(
        [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in date_ranges],
        key=lambda x: x[0],
    )

    # Merge overlapping intervals
    merged = []
    current_start, current_end = intervals[0]

    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged.append([current_start, current_end])
            current_start, current_end = start, end

    merged.append([current_start, current_end])

    # Calculate total days
    total_days = sum((end - start).days for start, end in merged)

    # Convert to months
    return total_days / 30.44


def compute_experience_at_start_ft(df):
    """Compute experience_at_start_ft excluding Pre-Entry positions.

    Args:
        df: DataFrame with experience data

    Returns:
        DataFrame with new column 'experience_at_start_ft'
    """
    df = df.copy()
    df["experience_at_start_ft"] = pd.Series(dtype=pd.Float32Dtype())

    for _prof_id, group in df.groupby("prof_id"):
        group_sorted = group.sort_values("exp_start_date")

        valid_mask = (
            group_sorted["exp_start_date"].notna()
            & group_sorted["exp_end_date"].notna()
        )
        valid_exps = group_sorted[valid_mask].copy()

        if len(valid_exps) == 0:
            continue

        for idx in valid_exps.index:
            current_start = df.loc[idx, "exp_start_date"]

            previous_exps = valid_exps[
                (valid_exps["exp_start_date"] < current_start)
                & (valid_exps["hierarchy_name"] != "Pre-Entry")
            ].copy()

            if len(previous_exps) == 0:
                df.loc[idx, "experience_at_start_ft"] = 0.0
                continue

            previous_exps["clipped_end"] = previous_exps["exp_end_date"].clip(
                upper=current_start
            )
            date_ranges = previous_exps[["exp_start_date", "clipped_end"]].to_numpy()
            total_months = calculate_non_overlapping_months(date_ranges)
            df.loc[idx, "experience_at_start_ft"] = round(total_months / 12, 2)

    return df


def apply_sample_selection_filters(df, tracking_log=None):
    """Apply sample selection filters and track removals.

    Args:
        df: DataFrame with experience data
        tracking_log: List to append tracking info to (optional)

    Returns:
        Filtered DataFrame
    """
    if tracking_log is None:
        tracking_log = []

    initial_count = len(df)
    tracking_log.append(
        {
            "step": "Initial data",
            "reason": "Loaded from file",
            "rows_before": 0,
            "rows_after": initial_count,
            "rows_removed": 0,
            "pct_removed": 0.0,
        }
    )

    # Apply filters sequentially
    filters = [
        (
            "Company filter",
            lambda x: x["comp_id"] != EXCLUDED_COMP_ID,
            f"Removed comp_id == {EXCLUDED_COMP_ID}",
        ),
        (
            "Job title filter",
            lambda x: (x["job_title"] != "") & (x["job_title"].notna()),
            "Empty or null job titles",
        ),
        (
            "Date filter",
            lambda x: (x["exp_start_date"] <= "2019-12-31")
            & (x["exp_end_date"] >= "2013-01-01"),
            "Outside 2013-2019 range",
        ),
        (
            "Company type filter",
            lambda x: (x["comp_type"] != "Self-Employed")
            & (x["comp_type"] != "Government Agency")
            & (x["comp_type"] != "Nonprofit")
            & (x["comp_type"] != "Educational Institution"),
            "Self-Employed, Government, Nonprofit, or Educational institutions",
        ),
        (
            "Location filter",
            lambda x: (x["match_method"] != "missing")
            & (x["match_method"] != "no_match"),
            "Missing or no location match",
        ),
        (
            "Hierarchy filter",
            lambda x: x["hierarchy_name"] != "Pre-Entry",
            "Pre-Entry positions",
        ),
        (
            "Company size filter (NA)",
            lambda x: x["min_size"].notna(),
            "Missing company size",
        ),
        (
            "Company size filter (==2)",
            lambda x: x["min_size"] != EXCLUDED_MIN_SIZE,
            f"Company size == {EXCLUDED_MIN_SIZE}",
        ),
    ]

    for filter_name, condition_func, reason in filters:
        rows_before = len(df)
        df = df[condition_func(df)]
        rows_after = len(df)
        rows_removed = rows_before - rows_after
        pct_removed = (rows_removed / rows_before * 100) if rows_before > 0 else 0

        tracking_log.append(
            {
                "step": filter_name,
                "reason": reason,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "rows_removed": rows_removed,
                "pct_removed": pct_removed,
            }
        )

    return df


def prepare_education_for_merge(education_data):
    """Prepare education data for merging with experience data.

    Args:
        education_data: DataFrame with education data

    Returns:
        Prepared education DataFrame with highest education so far
    """
    education_data["education_rank"] = education_data["education_grouped"].map(
        EDUCATION_HIERARCHY
    )

    educ_sorted = education_data.sort_values(
        ["prof_id", "ed_end_date", "education_rank"]
    )
    educ_sorted["cummax_education_rank"] = educ_sorted.groupby("prof_id")[
        "education_rank"
    ].cummax()
    educ_sorted["highest_education_so_far"] = educ_sorted["cummax_education_rank"].map(
        {v: k for k, v in EDUCATION_HIERARCHY.items()}
    )

    return educ_sorted[
        ["prof_id", "ed_id", "ed_start_date", "ed_end_date", "highest_education_so_far"]
    ].copy()


def merge_education_by_groups(exp_data, educ_data):
    """Merge education data with experience data using grouped merge_asof.

    This is a workaround for pandas merge_asof issues with large datasets.

    Args:
        exp_data: DataFrame with experience data
        educ_data: DataFrame with prepared education data

    Returns:
        Merged DataFrame
    """
    # Prepare data
    exp_sorted = exp_data[exp_data["exp_end_date"].notna()].copy()
    exp_sorted["prof_id"] = exp_sorted["prof_id"].astype("int64")

    educ_copy = educ_data[educ_data["ed_end_date"].notna()].copy()
    educ_copy["prof_id"] = educ_copy["prof_id"].astype("int64")

    # Sort
    exp_sorted = exp_sorted.sort_values(["prof_id", "exp_end_date"]).reset_index(
        drop=True
    )
    educ_copy = educ_copy.sort_values(["prof_id", "ed_end_date"]).reset_index(drop=True)

    # Find common prof_ids
    exp_prof_ids = set(exp_sorted["prof_id"].unique())
    educ_prof_ids = set(educ_copy["prof_id"].unique())
    common_prof_ids = sorted(exp_prof_ids & educ_prof_ids)

    # Merge by groups
    merged_groups = []
    for prof_id in common_prof_ids:
        exp_group = exp_sorted[exp_sorted["prof_id"] == prof_id].copy()
        educ_group = educ_copy[educ_copy["prof_id"] == prof_id].copy()

        merged_group = pd.merge_asof(
            exp_group,
            educ_group[
                ["ed_id", "ed_start_date", "ed_end_date", "highest_education_so_far"]
            ],
            left_on="exp_end_date",
            right_on="ed_end_date",
            direction="backward",
        )
        merged_groups.append(merged_group)

    # Concatenate all groups
    if merged_groups:
        merged = pd.concat(merged_groups, ignore_index=True)
    else:
        merged = exp_sorted.copy()
        merged["ed_id"] = pd.NA
        merged["ed_start_date"] = pd.NaT
        merged["ed_end_date"] = pd.NaT
        merged["highest_education_so_far"] = pd.NA

    # Add experiences without education data
    exp_without_educ = exp_sorted[~exp_sorted["prof_id"].isin(common_prof_ids)].copy()
    if len(exp_without_educ) > 0:
        exp_without_educ["ed_id"] = pd.NA
        exp_without_educ["ed_start_date"] = pd.NaT
        exp_without_educ["ed_end_date"] = pd.NaT
        exp_without_educ["highest_education_so_far"] = pd.NA
        merged = pd.concat([merged, exp_without_educ], ignore_index=True)

    merged = merged.rename(columns={"highest_education_so_far": "education_level"})

    return merged


def apply_education_filter(df, tracking_log=None):
    """Remove rows without education data and track removal.

    Args:
        df: DataFrame with merged education data
        tracking_log: List to append tracking info to (optional)

    Returns:
        Filtered DataFrame
    """
    if tracking_log is None:
        tracking_log = []

    rows_before = len(df)
    df = df[df["education_level"].notna()].copy()
    rows_after = len(df)
    rows_removed = rows_before - rows_after
    pct_removed = (rows_removed / rows_before * 100) if rows_before > 0 else 0

    tracking_log.append(
        {
            "step": "Education filter",
            "reason": "Missing education level",
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": rows_removed,
            "pct_removed": pct_removed,
        }
    )

    return df


def clean_company_names(df):
    """Clean company names by normalizing whitespace.

    Args:
        df: DataFrame with exp_company column

    Returns:
        DataFrame with cleaned company names
    """
    df = df.copy()
    df["exp_company"] = (
        df["exp_company"].str.replace(r"\s+", " ", regex=True).str.strip()
    )
    return df


def select_final_columns(df):
    """Select and order final columns for output.

    Args:
        df: DataFrame with all columns

    Returns:
        DataFrame with selected columns
    """
    columns_to_keep = [
        "exp_id",
        "prof_id",
        "comp_id",
        "industry_id",
        "job_title_id",
        "job_title",
        "exp_description",
        "exp_company",
        "industry",
        "matched_city",
        "matched_state",
        "bland_code",
        "match_method",
        "experience_at_start_recalc",
        "experience_at_start_ft",
        "exp_start_date",
        "exp_end_date",
        "duration",
        "date_reconstruction_method",
        "is_overlapping_reference",
        "gender",
        "hierarchy",
        "hierarchy_name",
        "comp_type",
        "min_size",
        "max_size",
        "total_size",
    ]

    # Only keep columns that exist
    existing_cols = [col for col in columns_to_keep if col in df.columns]
    return df[existing_cols]
