"""KNN-based selection of LinkedIn observations within SOEP-defined strata."""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def filter_to_double_sufficient_cells(
    soep_df,
    linkedin_df,
    cell_cols,
    min_soep_obs=5,
    min_linkedin_obs=5,
):
    """Filter both datasets to cells with sufficient observations.

    Args:
        soep_df: SOEP DataFrame with pgexpft and cell columns.
        linkedin_df: LinkedIn DataFrame with pgexpft and cell columns.
        cell_cols: List of columns defining cells (strata).
        min_soep_obs: Minimum SOEP observations per cell.
        min_linkedin_obs: Minimum LinkedIn observations per cell.

    Returns:
        Tuple of (filtered_soep_df, filtered_linkedin_df, sufficient_cells_df).
    """
    # Filter SOEP to cells with valid experience
    soep_valid = soep_df.dropna(subset=["pgexpft"]).copy()

    # Count SOEP observations per cell
    soep_cell_counts = (
        soep_valid.groupby(cell_cols).size().reset_index(name="soep_n_obs")
    )
    soep_sufficient = soep_cell_counts[
        soep_cell_counts["soep_n_obs"] >= min_soep_obs
    ].copy()

    # Filter SOEP to sufficient cells
    soep_filtered = soep_valid.merge(
        soep_sufficient[cell_cols], on=cell_cols, how="inner"
    )

    # Filter LinkedIn to matched observations only
    linkedin_matched = linkedin_df[linkedin_df["custom_id"].notna()].copy()
    linkedin_valid = linkedin_matched.dropna(subset=["pgexpft"])

    # Filter LinkedIn to cells that meet SOEP criteria
    linkedin_in_sufficient_cells = linkedin_valid.merge(
        soep_sufficient[cell_cols], on=cell_cols, how="inner"
    )

    # Count LinkedIn observations per cell
    linkedin_cell_counts = (
        linkedin_in_sufficient_cells.groupby(cell_cols)
        .size()
        .reset_index(name="linkedin_n_obs")
    )
    linkedin_sufficient = linkedin_cell_counts[
        linkedin_cell_counts["linkedin_n_obs"] >= min_linkedin_obs
    ].copy()

    # Get cells sufficient in both datasets
    double_sufficient_cells = linkedin_sufficient[cell_cols]

    # Re-filter both datasets to double-sufficient cells
    soep_final = soep_filtered.merge(
        double_sufficient_cells, on=cell_cols, how="inner"
    ).reset_index(drop=True)

    linkedin_final = linkedin_in_sufficient_cells.merge(
        double_sufficient_cells, on=cell_cols, how="inner"
    ).reset_index(drop=True)

    return soep_final, linkedin_final, double_sufficient_cells


def knn_select_within_cells(
    linkedin_df,
    soep_df,
    cell_cols,
    matches_per_soep=3,
    random_state=42,
):
    """Select LinkedIn observations using KNN matching within each cell.

    For each cell, selects n^{LI}_{s} = min(N^{LI}_{s}, c * N^{SOEP}_{s})
    observations where c = matches_per_soep.

    Args:
        linkedin_df: LinkedIn DataFrame with pgexpft column.
        soep_df: SOEP DataFrame with pgexpft column.
        cell_cols: List of columns defining cells (strata).
        matches_per_soep: Target ratio of LinkedIn to SOEP observations.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (selected_linkedin_df, selection_metadata_dict).
    """
    rng = np.random.default_rng(random_state)

    selected_dfs = []
    metadata = {
        "cells_processed": 0,
        "cells_all_selected": 0,
        "cells_knn_applied": 0,
        "cells_with_issues": 0,
        "total_soep": 0,
        "total_linkedin_before": 0,
        "total_linkedin_after": 0,
        "issue_log": [],
    }

    unique_cells = soep_df[cell_cols].drop_duplicates()

    for _, cell_row in unique_cells.iterrows():
        metadata["cells_processed"] += 1

        # Create cell mask
        cell_mask_soep = True
        cell_mask_linkedin = True
        for col in cell_cols:
            cell_mask_soep &= soep_df[col] == cell_row[col]
            cell_mask_linkedin &= linkedin_df[col] == cell_row[col]

        # Extract cell data
        soep_cell = soep_df[cell_mask_soep].copy()
        linkedin_cell = linkedin_df[cell_mask_linkedin].copy()

        # Remove missing pgexpft
        soep_cell = soep_cell[soep_cell["pgexpft"].notna()].copy()
        linkedin_cell = linkedin_cell[linkedin_cell["pgexpft"].notna()].copy()

        n_soep = len(soep_cell)
        n_linkedin = len(linkedin_cell)

        metadata["total_soep"] += n_soep
        metadata["total_linkedin_before"] += n_linkedin

        # Skip if insufficient data
        if n_soep == 0 or n_linkedin == 0:
            metadata["cells_with_issues"] += 1
            metadata["issue_log"].append(
                {
                    "cell": cell_row.to_dict(),
                    "reason": "no_data",
                    "soep_n": n_soep,
                    "linkedin_n": n_linkedin,
                }
            )
            continue

        # Calculate target sample size
        target_linkedin_size = min(n_linkedin, matches_per_soep * n_soep)

        # Case 1: Select all LinkedIn observations
        if n_linkedin <= matches_per_soep * n_soep:
            selected_dfs.append(linkedin_cell)
            metadata["total_linkedin_after"] += n_linkedin
            metadata["cells_all_selected"] += 1

        # Case 2: Apply KNN to select subset
        else:
            soep_exp = soep_cell[["pgexpft"]].to_numpy()
            linkedin_exp = linkedin_cell[["pgexpft"]].to_numpy()

            # Fit KNN model on SOEP observations
            knn = NearestNeighbors(
                n_neighbors=min(n_linkedin, matches_per_soep), metric="euclidean"
            )
            knn.fit(linkedin_exp)

            # Find k nearest LinkedIn neighbors for each SOEP observation
            _, indices = knn.kneighbors(soep_exp)

            # Flatten indices to get all matched LinkedIn observations
            matched_linkedin_indices = np.unique(indices.flatten())

            # If more matches than needed, sample randomly
            if len(matched_linkedin_indices) > target_linkedin_size:
                matched_linkedin_indices = rng.choice(
                    matched_linkedin_indices, size=target_linkedin_size, replace=False
                )

            # Select the LinkedIn observations
            selected_linkedin = linkedin_cell.iloc[matched_linkedin_indices].copy()
            selected_dfs.append(selected_linkedin)
            metadata["total_linkedin_after"] += len(selected_linkedin)
            metadata["cells_knn_applied"] += 1

    # Concatenate all selected cells
    if selected_dfs:
        linkedin_selected = pd.concat(selected_dfs, ignore_index=True)
    else:
        linkedin_selected = pd.DataFrame(columns=linkedin_df.columns)

    return linkedin_selected, metadata
