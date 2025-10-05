"""Functions to clean SQL data sets."""

import numpy as np
import pandas as pd


def clean_experience_data(experience_df: pd.DataFrame) -> pd.DataFrame:
    """Clean the experience data set.

    Args:
        experience_df (pd.DataFrame): The raw experience data set.

    Returns:
        pd.DataFrame: The cleaned experience data set.
    """
    clean_df = pd.DataFrame()

    clean_df["exp_id"] = experience_df["exp_id"].astype(pd.Int32Dtype())
    clean_df["prof_id"] = experience_df["prof_id"].astype(pd.Int32Dtype())
    clean_df["comp_id"] = experience_df["comp_id"].astype(pd.Int32Dtype())
    clean_df["industry_id"] = experience_df["industry_id"].astype(pd.Int32Dtype())
    clean_df["job_title_id"] = experience_df["job_title_id"].astype(pd.Int32Dtype())

    clean_df["job_title"] = experience_df["job_title"]
    clean_df["job_title_cleaned"] = experience_df["job_title_cleaned"]
    clean_df["job_title_standard"] = experience_df["job_title_standard"]
    clean_df["exp_description"] = experience_df["exp_description"]

    clean_df["experience_at_start"] = experience_df["total_experience"].astype(
        pd.Float32Dtype()
    )
    clean_df["duration"] = experience_df["duration"].astype(pd.Float32Dtype())
    clean_df["exp_start_date"] = pd.to_datetime(
        experience_df["exp_start_date"], errors="coerce"
    )
    clean_df["exp_end_date"] = pd.to_datetime(
        experience_df["exp_end_date"], errors="coerce"
    )
    clean_df["exp_present"] = experience_df["present"].astype(pd.BooleanDtype())
    clean_df["is_last_experience"] = experience_df["is_last_experience"].astype(
        pd.BooleanDtype()
    )

    clean_df["hierarchy"] = experience_df["hierarchy"].astype(pd.Int8Dtype())
    clean_df["hierarchy_name"] = experience_df["hierarchy_name"].astype(
        pd.CategoricalDtype()
    )
    clean_df["gender"] = (
        experience_df["gender"].replace({"F": 1, "M": 0}).astype(pd.Int8Dtype())
    )
    clean_df["prof_location"] = experience_df["prof_location"]
    clean_df["prof_city"] = experience_df["prof_city"]
    clean_df["prof_state"] = experience_df["prof_state"]
    clean_df["prof_country"] = experience_df["prof_country"]
    clean_df["prof_industry"] = experience_df["prof_industry"].astype(
        pd.CategoricalDtype()
    )
    clean_df["industry"] = experience_df["industry"].astype(pd.CategoricalDtype())
    clean_df["crawling_date"] = pd.to_datetime(
        experience_df["crawling_date"], errors="coerce"
    )

    clean_df["exp_company"] = experience_df["exp_company"]
    clean_df["company"] = experience_df["company"].where(
        experience_df["company"] != "deutschsprachige Theater", pd.NA
    )
    clean_df["comp_type"] = experience_df["comp_type"].astype(pd.CategoricalDtype())
    clean_df["min_size"] = experience_df["min_size"].astype(pd.Int32Dtype())
    clean_df["max_size"] = experience_df["max_size"].astype(pd.Int32Dtype())
    clean_df["total_size"] = experience_df["total_size"].astype(pd.Int32Dtype())
    clean_df["exp_location"] = experience_df["exp_location"]
    clean_df["comp_location"] = experience_df["comp_location"]
    clean_df["comp_city"] = experience_df["comp_city"]
    clean_df["comp_postal_code"] = experience_df["comp_postal_code"]
    clean_df["comp_address"] = experience_df["comp_address"]
    clean_df["comp_headquarter"] = experience_df["comp_headquarter"]
    clean_df["top_400"] = experience_df["top_400"].astype(pd.BooleanDtype())
    clean_df["founded"] = experience_df["founded"].astype(pd.Int64Dtype())
    clean_df["followers_on_linkedin"] = experience_df["followers_on_linkedin"].astype(
        pd.Int64Dtype()
    )

    # Sort and clean dates efficiently
    clean_df = clean_df.sort_values(
        by=["prof_id", "experience_at_start"]
    )  # update this sort after cleaning function is complete
    # cleaned_df = clean_dates(sorted_df)

    return clean_df


def clean_education_data(education_df: pd.DataFrame) -> pd.DataFrame:
    return education_df


def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Efficiently clean and reconstruct experience dates using vectorized operations.

    Args:
        df (pd.DataFrame): Sorted DataFrame with experience data

    Returns:
        pd.DataFrame: DataFrame with cleaned dates
    """
    # Work on a copy to avoid modifying the original
    result_df = df.copy()

    # Add experience_at_end column vectorized
    result_df["experience_at_end"] = result_df.groupby("prof_id")[
        "experience_at_start"
    ].shift(-1)

    # Step 1: Identify and clear duplicate dates vectorized
    result_df = _identify_and_clear_duplicates(result_df)

    # Step 2: Add reference columns for efficient processing
    result_df = _add_reference_columns(result_df)

    # Step 3: Reconstruct dates using vectorized operations
    result_df = _reconstruct_dates_vectorized(result_df)

    # Clean up temporary columns
    cols_to_drop = [
        "experience_at_end",
        "has_missing_dates",
        "prev_valid_end",
        "next_valid_start",
        "prev_valid_end_exp",
        "next_valid_start_exp",
        "position_in_group",
    ]
    result_df = result_df.drop(
        columns=[col for col in cols_to_drop if col in result_df.columns]
    )

    return result_df


def _identify_and_clear_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized identification and clearing of duplicate dates."""
    # Find experiences for a profile with same company and start and end dates
    duplicate_mask = (
        (
            df.groupby(["prof_id", "exp_company", "exp_start_date", "exp_end_date"])[
                "exp_id"
            ].transform("count")
            > 1
        )
        & df["exp_start_date"].notna()
        & df["exp_end_date"].notna()
    )

    # Clear duplicates in one operation
    df.loc[duplicate_mask, ["exp_start_date", "exp_end_date"]] = pd.NaT

    # Mark rows that need reconstruction
    df["has_missing_dates"] = df["exp_start_date"].isna() | df["exp_end_date"].isna()

    return df


def _add_reference_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add reference columns for previous and next valid dates using
    vectorized operations.
    """  # noqa: D205
    df_sorted = df.copy()

    # Create mask for valid dates
    valid_dates_mask = (
        df_sorted["exp_start_date"].notna() & df_sorted["exp_end_date"].notna()
    )

    # Previous valid references
    df_sorted["prev_valid_end"] = np.where(
        valid_dates_mask, df_sorted["exp_end_date"], pd.NaT
    )
    df_sorted["prev_valid_end"] = df_sorted.groupby("prof_id")["prev_valid_end"].fillna(
        method="ffill"
    )

    df_sorted["prev_valid_end_exp"] = np.where(
        valid_dates_mask, df_sorted["experience_at_end"], np.nan
    )
    df_sorted["prev_valid_end_exp"] = df_sorted.groupby("prof_id")[
        "prev_valid_end_exp"
    ].fillna(method="ffill")

    # Next valid references
    df_sorted["next_valid_start"] = np.where(
        valid_dates_mask, df_sorted["exp_start_date"], pd.NaT
    )
    df_sorted["next_valid_start"] = df_sorted.groupby("prof_id")[
        "next_valid_start"
    ].fillna(method="bfill")

    df_sorted["next_valid_start_exp"] = np.where(
        valid_dates_mask, df_sorted["experience_at_start"], np.nan
    )
    df_sorted["next_valid_start_exp"] = df_sorted.groupby("prof_id")[
        "next_valid_start_exp"
    ].fillna(method="bfill")

    # Add position within groups of missing dates for sequential processing
    df_sorted["position_in_group"] = df_sorted.groupby(
        ["prof_id", "has_missing_dates"]
    ).cumcount()

    return df_sorted


def _reconstruct_dates_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct missing dates using vectorized operations where possible."""
    missing_mask = df["has_missing_dates"]

    if not missing_mask.any():
        return df

    # Calculate experience differences vectorized - handle NA values properly
    exp_diff = df["experience_at_end"] - df["experience_at_start"]

    # Create months_duration with proper NA handling
    valid_exp_diff = exp_diff.notna() & (exp_diff > 0)
    months_duration = pd.Series(index=df.index, dtype="Int64")  # Use nullable integer

    # Fill valid values
    months_duration.loc[valid_exp_diff] = np.clip(
        (exp_diff.loc[valid_exp_diff] * 12).astype(pd.Int64Dtype()), 1, None
    )

    # Fill invalid/missing values with default
    months_duration = months_duration.fillna(3)

    # Calculate spans between references
    time_spans = _calculate_time_spans(df[missing_mask])
    exp_spans = _calculate_experience_spans(df[missing_mask])

    # Determine reconstruction method vectorized
    concurrent_mask = (exp_spans * 12) > time_spans

    # Reconstruct dates based on method
    df = _apply_reconstruction_methods(
        df, missing_mask, concurrent_mask, months_duration
    )

    return df


def _calculate_time_spans(df_missing: pd.DataFrame) -> pd.Series:
    """Calculate time spans between reference dates in months."""
    time_diff = df_missing["next_valid_start"] - df_missing["prev_valid_end"]
    months_span = (
        (
            time_diff.dt.days / 30.44  # Average days per month
        )
        .fillna(12)
        .clip(lower=1)
    )

    return months_span


def _calculate_experience_spans(df_missing: pd.DataFrame) -> pd.Series:
    """Calculate experience spans between reference points."""
    exp_spans = (
        (df_missing["next_valid_start_exp"] - df_missing["prev_valid_end_exp"])
        .fillna(0.25)
        .clip(lower=0.25)
    )

    return exp_spans


def _apply_reconstruction_methods(
    df: pd.DataFrame,
    missing_mask: pd.Series,
    concurrent_mask: pd.Series,
    months_duration: pd.Series,
) -> pd.DataFrame:
    """Apply reconstruction methods efficiently using vectorized operations
    where possible.
    """  # noqa: D205
    has_prev = df["prev_valid_end"].notna()
    has_next = df["next_valid_start"].notna()

    prev_only_mask = missing_mask & has_prev & ~has_next
    next_only_mask = missing_mask & ~has_prev & has_next
    both_mask = missing_mask & has_prev & has_next

    # Reconstruct prev_only cases vectorized
    if prev_only_mask.any():
        df = _reconstruct_prev_only_vectorized(df, prev_only_mask, months_duration)

    # Reconstruct next_only cases vectorized
    if next_only_mask.any():
        df = _reconstruct_next_only_vectorized(df, next_only_mask, months_duration)

    # For both_references cases, still need sequential processing per professional
    # but optimized with better data structures
    if both_mask.any():
        df = _reconstruct_both_references_optimized(df, both_mask, concurrent_mask)

    return df


def _reconstruct_prev_only_vectorized(
    df: pd.DataFrame, mask: pd.Series, months_duration: pd.Series
) -> pd.DataFrame:
    """Vectorized reconstruction for prev_only cases."""
    df.loc[mask, "cumsum_months"] = (
        df[mask]
        .groupby("prof_id")
        .apply(
            lambda x: x.reset_index()["position_in_group"].apply(
                lambda pos: (months_duration[x.index] + 1).iloc[: pos + 1].sum()
            )
        )
        .to_numpy()
    )

    # Calculate start dates
    start_dates = (
        df.loc[mask, "prev_valid_end"]
        + pd.to_timedelta(df.loc[mask, "cumsum_months"], unit="D") * 30.44
    )
    end_dates = start_dates + pd.to_timedelta(months_duration[mask], unit="D") * 30.44

    df.loc[mask, "exp_start_date"] = start_dates
    df.loc[mask, "exp_end_date"] = end_dates
    df.loc[mask, "duration"] = months_duration[mask] / 12
    df.loc[mask, "date_reconstruction_method"] = "from_previous"

    return df


def _reconstruct_next_only_vectorized(
    df: pd.DataFrame, mask: pd.Series, months_duration: pd.Series
) -> pd.DataFrame:
    """Vectorized reconstruction for next_only cases."""
    df_subset = df[mask].copy()
    df_subset["reverse_pos"] = (
        df_subset.groupby("prof_id")["position_in_group"].transform("max")
        - df_subset["position_in_group"]
    )

    # Calculate cumulative months working backwards
    cumsum_months = (
        df_subset.groupby("prof_id")
        .apply(
            lambda x: x.sort_values("reverse_pos")
            .reset_index()
            .apply(
                lambda row: (months_duration[x.index] + 1)
                .iloc[: int(row["reverse_pos"]) + 1]
                .sum(),
                axis=1,
            )
        )
        .to_numpy()
    )

    # Calculate end dates working backwards
    end_dates = (
        df.loc[mask, "next_valid_start"]
        - pd.to_timedelta(cumsum_months, unit="D") * 30.44
    )
    start_dates = end_dates - pd.to_timedelta(months_duration[mask], unit="D") * 30.44

    df.loc[mask, "exp_start_date"] = start_dates
    df.loc[mask, "exp_end_date"] = end_dates
    df.loc[mask, "duration"] = months_duration[mask] / 12
    df.loc[mask, "date_reconstruction_method"] = "from_next"

    return df


def _reconstruct_both_references_optimized(
    df: pd.DataFrame, mask: pd.Series, concurrent_mask: pd.Series
) -> pd.DataFrame:
    """Optimized reconstruction for both_references cases.
    Uses efficient data structures and minimal loops.
    """  # noqa: D205
    both_subset = df[mask].copy()

    # Pre-calculate all needed values
    both_subset["exp_span"] = (
        both_subset["next_valid_start_exp"] - both_subset["prev_valid_end_exp"]
    )
    both_subset["time_span_months"] = _calculate_time_spans(both_subset)
    both_subset["role_exp_span"] = (
        both_subset["experience_at_end"] - both_subset["experience_at_start"]
    )
    both_subset["is_concurrent"] = concurrent_mask[mask]

    # Process each professional's data efficiently
    results = []
    for _prof_id, group in both_subset.groupby("prof_id"):
        group_result = _process_professional_both_refs(group)
        results.append(group_result)

    if results:
        result_df = pd.concat(results)

        # Update original dataframe
        df.loc[result_df.index, "exp_start_date"] = result_df["exp_start_date"]
        df.loc[result_df.index, "exp_end_date"] = result_df["exp_end_date"]
        df.loc[result_df.index, "duration"] = result_df["duration"]
        df.loc[result_df.index, "date_reconstruction_method"] = result_df[
            "date_reconstruction_method"
        ]

    return df


def _process_professional_both_refs(group: pd.DataFrame) -> pd.DataFrame:
    """Process a single professional's both_references positions efficiently."""
    group = group.sort_values("experience_at_start").copy()

    # Initialize with first reference
    prev_end = group.iloc[0]["prev_valid_end"]

    for idx, row in group.iterrows():
        # Calculate position dates
        start_date = prev_end + pd.DateOffset(months=1)

        # Calculate duration based on method
        if row["is_concurrent"]:
            role_portion = (
                row["role_exp_span"] / row["exp_span"] if row["exp_span"] > 0 else 0.5
            )
            role_months = max(3, int(row["time_span_months"] * role_portion))
            method = "from_both_concurrent"
        else:
            role_months = (
                max(3, int(12 * row["role_exp_span"]))
                if pd.notna(row["role_exp_span"])
                else 3
            )
            method = "from_both"

        end_date = start_date + pd.DateOffset(months=role_months)

        # Ensure doesn't exceed next reference
        if end_date >= row["next_valid_start"]:
            end_date = row["next_valid_start"] - pd.DateOffset(months=1)

        # Update row
        group.loc[idx, "exp_start_date"] = start_date
        group.loc[idx, "exp_end_date"] = end_date
        group.loc[idx, "duration"] = role_months / 12
        group.loc[idx, "date_reconstruction_method"] = method

        # Update for next iteration
        prev_end = end_date

    return group
