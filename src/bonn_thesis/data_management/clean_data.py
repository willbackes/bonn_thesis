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
    clean_df["gender"] = experience_df["gender"].astype(pd.CategoricalDtype())
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
    clean_df = clean_dates(clean_df)

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

    result_df = result_df.sort_values(by=["prof_id", "experience_at_start"])

    # Add experience_at_end column vectorized
    result_df["experience_at_end"] = result_df.groupby("prof_id")[
        "experience_at_start"
    ].shift(-1)

    # Step 1: Identify and clear duplicate dates vectorized
    result_df = _identify_and_clear_duplicates(result_df)

    # Step 2: Identify overlapping experiences
    result_df = _identify_overlapping_experiences(result_df)

    # Step 3: Add reference columns for efficient processing
    result_df = _add_reference_columns(result_df)

    # Step 4: Reconstruct dates based on the new method
    result_df = _reconstruct_dates(result_df)

    # Clean up temporary columns
    cols_to_drop = [
        "experience_at_end",
        "has_missing_dates",
        "prev_valid_end",
        "next_valid_start",
        "prev_valid_end_exp",
        "next_valid_start_exp",
        "position_in_group",
        # "is_overlapping_reference",
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


def _identify_overlapping_experiences(df: pd.DataFrame) -> pd.DataFrame:
    """Identify overlapping experiences and flag shorter ones that shouldn't
    be used as references.

    An experience is flagged if:
    - It has valid dates
    - It overlaps with another experience in a different company
    - It starts after AND ends before the overlapping experience
    """
    df["is_overlapping_reference"] = False

    for _prof_id, group in df.groupby("prof_id"):
        # Only consider experiences with valid dates
        valid_dates = group[
            group["exp_start_date"].notna() & group["exp_end_date"].notna()
        ].copy()

        if len(valid_dates) <= 1:
            continue

        # Compare each experience with all others
        for i, (idx_i, exp_i) in enumerate(valid_dates.iterrows()):
            for j, (idx_j, exp_j) in enumerate(valid_dates.iterrows()):
                if i >= j:  # Skip self-comparison and avoid duplicate checks
                    continue

                # Check if they're from different companies
                if exp_i["exp_company"] == exp_j["exp_company"]:
                    continue

                # Check for overlap
                overlap_start = max(exp_i["exp_start_date"], exp_j["exp_start_date"])
                overlap_end = min(exp_i["exp_end_date"], exp_j["exp_end_date"])

                if overlap_start < overlap_end:
                    # There is an overlap

                    # exp_i is contained within exp_j
                    if (
                        exp_i["exp_start_date"] >= exp_j["exp_start_date"]
                        and exp_i["exp_end_date"] <= exp_j["exp_end_date"]
                    ):
                        df.loc[idx_i, "is_overlapping_reference"] = True

                    # exp_j is contained within exp_i
                    elif (
                        exp_j["exp_start_date"] >= exp_i["exp_start_date"]
                        and exp_j["exp_end_date"] <= exp_i["exp_end_date"]
                    ):
                        df.loc[idx_j, "is_overlapping_reference"] = True

    return df


def _add_reference_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add reference columns for previous and next valid dates using vectorized
    operations.

    Modified to skip experiences flagged as overlapping references.
    """
    df_sorted = df.copy()

    # Create mask for valid dates that are NOT flagged as overlapping references
    valid_dates_mask = (
        df_sorted["exp_start_date"].notna()
        & df_sorted["exp_end_date"].notna()
        & ~df_sorted["is_overlapping_reference"]
    )

    # Previous valid references - initialize as NaT datetime series
    df_sorted["prev_valid_end"] = pd.NaT
    df_sorted["prev_valid_end"] = df_sorted["prev_valid_end"].astype("datetime64[ns]")
    df_sorted.loc[valid_dates_mask, "prev_valid_end"] = df_sorted.loc[
        valid_dates_mask, "exp_end_date"
    ]
    df_sorted["prev_valid_end"] = df_sorted.groupby("prof_id")["prev_valid_end"].ffill()

    df_sorted["prev_valid_end_exp"] = pd.Series(dtype=pd.Float32Dtype())
    df_sorted.loc[valid_dates_mask, "prev_valid_end_exp"] = df_sorted.loc[
        valid_dates_mask, "experience_at_end"
    ]
    df_sorted["prev_valid_end_exp"] = df_sorted.groupby("prof_id")[
        "prev_valid_end_exp"
    ].ffill()

    # Next valid references - initialize as NaT datetime series
    df_sorted["next_valid_start"] = pd.NaT
    df_sorted["next_valid_start"] = df_sorted["next_valid_start"].astype(
        "datetime64[ns]"
    )
    df_sorted.loc[valid_dates_mask, "next_valid_start"] = df_sorted.loc[
        valid_dates_mask, "exp_start_date"
    ]
    df_sorted["next_valid_start"] = df_sorted.groupby("prof_id")[
        "next_valid_start"
    ].bfill()

    df_sorted["next_valid_start_exp"] = pd.Series(dtype=pd.Float32Dtype())
    df_sorted.loc[valid_dates_mask, "next_valid_start_exp"] = df_sorted.loc[
        valid_dates_mask, "experience_at_start"
    ]
    df_sorted["next_valid_start_exp"] = df_sorted.groupby("prof_id")[
        "next_valid_start_exp"
    ].bfill()

    # Add position within consecutive sequences of missing dates
    df_sorted["position_in_group"] = 0

    # For each profile, identify consecutive sequences of missing dates
    for _prof_id, group in df_sorted.groupby("prof_id"):
        mask = group["has_missing_dates"]

        # Create sequence groups: increment counter when transitioning False to True
        sequence_change = mask & ~mask.shift(1, fill_value=False)
        sequence_id = sequence_change.cumsum()

        # Count within each sequence, but only for True values
        position = np.where(mask, group.groupby(sequence_id).cumcount() + 1, 0)

        df_sorted.loc[group.index, "position_in_group"] = position

    return df_sorted


def _reconstruct_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct missing dates based on available reference points.

    Identifies 5 different cases and applies appropriate reconstruction method.
    """
    missing_mask = df["has_missing_dates"]

    if not missing_mask.any():
        return df

    # Initialize date_reconstruction_method column
    df["date_reconstruction_method"] = None

    # Calculate date differences in months - handle NaT values properly
    # First compute the timedelta
    date_diff = df["next_valid_start"] - df["prev_valid_end"]

    # Convert to days, handling NaT by checking if the series is datetime-like
    if pd.api.types.is_timedelta64_dtype(date_diff):
        df["diff_date"] = date_diff.dt.days / 30.44
    else:
        # If subtraction resulted in object dtype due to all NaT, create NaN series
        df["diff_date"] = pd.Series([np.nan] * len(df), index=df.index)

    # Calculate experience differences (already in years, convert to months)
    df["diff_exp"] = (df["next_valid_start_exp"] - df["prev_valid_end_exp"]) * 12

    # Identify the 5 cases
    has_prev = df["prev_valid_end"].notna()
    has_next = df["next_valid_start"].notna()

    # Case 1: Only previous reference available
    case_1_mask = missing_mask & has_prev & ~has_next

    # Case 2: Only next reference available
    case_2_mask = missing_mask & ~has_prev & has_next

    # Case 3 & 4: Both references available
    both_refs_mask = missing_mask & has_prev & has_next

    # For case 3 and 4, we need both diff values to be valid
    valid_diffs_mask = both_refs_mask & df["diff_exp"].notna() & df["diff_date"].notna()

    case_3_mask = valid_diffs_mask & (df["diff_exp"] > df["diff_date"])
    case_4_mask = valid_diffs_mask & (df["diff_exp"] <= df["diff_date"])

    # Case 5: No references available
    case_5_mask = missing_mask & ~has_prev & ~has_next

    # Apply reconstruction methods
    if case_1_mask.any():
        df = _previous_only(df, case_1_mask)

    if case_2_mask.any():
        df = _next_only(df, case_2_mask)

    if case_3_mask.any():
        df = _both_no_inactive_period(df, case_3_mask)

    if case_4_mask.any():
        df = _both_with_inactive_period(df, case_4_mask)

    if case_5_mask.any():
        df.loc[case_5_mask, "date_reconstruction_method"] = "no_reference"

    # Clean up temporary columns
    df = df.drop(columns=["diff_date", "diff_exp"], errors="ignore")

    return df


def _previous_only(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Reconstruct dates when only previous reference is available.

    Use the latest prev_valid_end across the whole profile as the starting reference.
    """
    for prof_id, group in df[mask].groupby("prof_id"):
        sorted_group = group.sort_values("experience_at_start")

        # Use the latest prev_valid_end available for the profile
        profile_prev = df.loc[df["prof_id"] == prof_id, "prev_valid_end"].dropna()
        if len(profile_prev) == 0:
            # no profile-level previous reference -> skip
            continue
        prev_end = pd.to_datetime(profile_prev.max())

        for idx in sorted_group.index:
            exp_start = prev_end + pd.DateOffset(months=1)

            exp_diff = (
                df.loc[idx, "experience_at_end"] - df.loc[idx, "experience_at_start"]
            )
            if pd.notna(exp_diff) and exp_diff > 0:
                duration_months = int(exp_diff * 12)
            else:
                duration_months = 3

            exp_end = exp_start + pd.DateOffset(months=duration_months)

            df.loc[idx, "exp_start_date"] = exp_start
            df.loc[idx, "exp_end_date"] = exp_end
            df.loc[idx, "duration"] = duration_months / 12
            df.loc[idx, "date_reconstruction_method"] = "previous_only"

            prev_end = exp_end

    return df


def _next_only(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Reconstruct dates when only next reference is available.

    Use the earliest next_valid_start available for the profile as the reference.
    """
    for prof_id, group in df[mask].groupby("prof_id"):
        sorted_group = group.sort_values("experience_at_start", ascending=False)

        # Use the earliest next_valid_start available for the profile
        profile_next = df.loc[df["prof_id"] == prof_id, "next_valid_start"].dropna()
        if len(profile_next) == 0:
            # no profile-level next reference -> skip
            continue
        next_start = pd.to_datetime(profile_next.min())

        for idx in sorted_group.index:
            exp_end = next_start - pd.DateOffset(months=1)

            exp_diff = (
                df.loc[idx, "experience_at_end"] - df.loc[idx, "experience_at_start"]
            )
            if pd.notna(exp_diff) and exp_diff > 0:
                duration_months = int(exp_diff * 12)
            else:
                duration_months = 3

            exp_start = exp_end - pd.DateOffset(months=duration_months)

            df.loc[idx, "exp_start_date"] = exp_start
            df.loc[idx, "exp_end_date"] = exp_end
            df.loc[idx, "duration"] = duration_months / 12
            df.loc[idx, "date_reconstruction_method"] = "next_only"

            next_start = exp_start

    return df


def _both_no_inactive_period(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Reconstruct dates when both references available and diff_exp > diff_date.

    Accommodates longer experience span into available date span (concurrent roles).
    """
    # Process each profile group separately
    for _prof_id, group in df[mask].groupby("prof_id"):
        sorted_group = group.sort_values("experience_at_start")

        prev_end = pd.Timestamp(sorted_group.iloc[0]["prev_valid_end"])
        prev_valid_end_exp = float(sorted_group.iloc[0]["prev_valid_end_exp"])
        next_valid_start_exp = float(sorted_group.iloc[0]["next_valid_start_exp"])

        # Calculate total experience span
        total_exp_span = next_valid_start_exp - prev_valid_end_exp

        for idx in sorted_group.index:
            # Start date: 1 month after previous end
            exp_start = prev_end + pd.DateOffset(months=1)

            # Duration based on proportional experience
            exp_diff = (
                df.loc[idx, "experience_at_end"] - df.loc[idx, "experience_at_start"]
            )

            if pd.notna(exp_diff) and exp_diff > 0 and total_exp_span > 0:
                # Proportional allocation
                proportion = exp_diff / total_exp_span
                available_months = float(df.loc[idx, "diff_date"])
                duration_months = max(3, int(available_months * proportion))
            else:
                duration_months = 3  # Default minimum for zero experience

            exp_end = exp_start + pd.DateOffset(months=duration_months)

            # Update dataframe
            df.loc[idx, "exp_start_date"] = exp_start
            df.loc[idx, "exp_end_date"] = exp_end
            df.loc[idx, "duration"] = duration_months / 12
            df.loc[idx, "date_reconstruction_method"] = "both_no_inactive"

            # Update for next iteration
            prev_end = exp_end

    return df


def _both_with_inactive_period(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Reconstruct dates when both references available and diff_exp <= diff_date.

    Distributes unemployment periods uniformly between job transitions.
    """
    # Process each profile group separately
    for _prof_id, group in df[mask].groupby("prof_id"):
        sorted_group = group.sort_values("experience_at_start")

        prev_end = pd.Timestamp(sorted_group.iloc[0]["prev_valid_end"])

        # Calculate inactive period per transition
        n = len(sorted_group)
        if n > 1:
            total_inactive = float(df.loc[sorted_group.index[0], "diff_date"]) - float(
                df.loc[sorted_group.index[0], "diff_exp"]
            )
            inactive_per_transition = (
                total_inactive / (n - 1) if total_inactive > 0 else 0
            )
        else:
            inactive_per_transition = 0

        for i, idx in enumerate(sorted_group.index):
            # Start date: 1 month after previous end
            exp_start = prev_end + pd.DateOffset(months=1)

            # Duration based on experience difference
            exp_diff = (
                df.loc[idx, "experience_at_end"] - df.loc[idx, "experience_at_start"]
            )
            if pd.notna(exp_diff) and exp_diff > 0:
                duration_months = int(exp_diff * 12)
            else:
                duration_months = 3  # Default minimum

            exp_end = exp_start + pd.DateOffset(months=duration_months)

            # Update dataframe
            df.loc[idx, "exp_start_date"] = exp_start
            df.loc[idx, "exp_end_date"] = exp_end
            df.loc[idx, "duration"] = duration_months / 12
            df.loc[idx, "date_reconstruction_method"] = "both_with_inactive"

            # Update for next iteration (add inactive period if not last experience)
            if i < n - 1:
                prev_end = exp_end + pd.DateOffset(months=int(inactive_per_transition))
            else:
                prev_end = exp_end

    return df
