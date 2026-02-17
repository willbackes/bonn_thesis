"""Define strata for sampling based on SOEP cell structure."""


def create_cell_lookup(soep_cells_df):
    """Create lookup dictionary mapping cell characteristics to custom_id and n_obs.

    Args:
        soep_cells_df: DataFrame with SOEP cell structure containing unique
            combinations of: syear, isco_3_name, education_grouped, sex_en,
            state_en, custom_id, n_obs

    Returns:
        Dictionary: {(syear, isco_3_name, education_grouped, sex_en, state_en):
                     {'custom_id': ..., 'n_obs': ...}}
    """
    cell_lookup = {}

    for _, row in soep_cells_df.iterrows():
        key = (
            row["syear"],
            row["isco_3_name"],
            row["education_grouped"],
            row["sex_en"],
            row["state_en"],
        )
        cell_lookup[key] = {"custom_id": row["custom_id"], "n_obs": row["n_obs"]}

    return cell_lookup
