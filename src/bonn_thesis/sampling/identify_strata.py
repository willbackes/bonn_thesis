"""Assign custom_id and n_obs to LinkedIn observations based on SOEP cell metadata."""


def assign_custom_id_to_linkedin(linkedin_df, cell_lookup):
    """Assign custom_id and n_obs to LinkedIn observations.

    Args:
        linkedin_df: DataFrame with LinkedIn data containing:
            - syear, isco_3_name, education_grouped, sex_en, state_en
        cell_lookup: Dictionary mapping cell characteristics to metadata dict

    Returns:
        DataFrame: LinkedIn data with custom_id and soep_n_obs columns added
    """
    result_df = linkedin_df.copy()

    # Create cell key and lookup metadata for each observation
    metadata = result_df.apply(
        lambda row: cell_lookup.get(
            (
                row["syear"],
                row["isco_3_name"],
                row["education_grouped"],
                row["sex_en"],
                row["state_en"],
            ),
            {"custom_id": None, "n_obs": None},
        ),
        axis=1,
        result_type="expand",
    )

    result_df["custom_id"] = metadata["custom_id"]
    result_df["soep_n_obs"] = metadata["n_obs"]

    return result_df
