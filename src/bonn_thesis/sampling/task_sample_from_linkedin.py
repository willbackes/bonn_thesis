"""Tasks for KNN-based sampling of LinkedIn data within SOEP-defined strata."""

import json
from pathlib import Path

import pandas as pd
from pytask import task

from bonn_thesis.config import (
    LINKEDIN_MATCHED_TO_SOEP_BLD,
    MATCHES_PER_SOEP,
    MERGED_EXP_ED_SAMPLING_BLD,
    MIN_LINKEDIN_OBS,
    MIN_SOEP_OBS,
    RANDOM_STATE,
    SOEP_DATA_BLD,
    STRATA_COLS,
)
from bonn_thesis.sampling.sample_linkedin_match_soep import (
    filter_to_double_sufficient_cells,
    knn_select_within_cells,
)


@task(
    id="sample_linkedin_knn",
    kwargs={
        "soep_file": SOEP_DATA_BLD / "soep_clean.parquet",
        "produces": {
            "data": LINKEDIN_MATCHED_TO_SOEP_BLD / "linkedin_selected.parquet",
            "metadata": LINKEDIN_MATCHED_TO_SOEP_BLD / "linkedin_knn_metadata.jsonl",
        },
    },
)
def task_sample_linkedin_knn(
    soep_file: Path,
    produces: dict,
) -> None:
    """Sample LinkedIn observations using KNN matching within SOEP strata.

    Args:
        soep_file: Path to cleaned SOEP data.
        produces: Dictionary with paths for data and metadata outputs.
    """
    # Ensure output directory exists
    produces["data"].parent.mkdir(parents=True, exist_ok=True)

    # Load SOEP data
    soep_all = pd.read_parquet(soep_file, engine="fastparquet")
    soep_all = soep_all.dropna(subset=["pgexpft"])

    # Load all LinkedIn strata files
    linkedin_files = sorted(
        MERGED_EXP_ED_SAMPLING_BLD.glob("linkedin_merged_exp_ed_strata_*.parquet")
    )

    if not linkedin_files:
        # Save empty results if no files found
        pd.DataFrame().to_parquet(produces["data"], engine="fastparquet", index=False)
        with produces["metadata"].open("w") as f:
            json.dump({"error": "No LinkedIn strata files found"}, f, indent=2)
        return

    linkedin_dfs = []
    for file in linkedin_files:
        df = pd.read_parquet(file, engine="fastparquet")
        linkedin_dfs.append(df)

    linkedin_all = pd.concat(linkedin_dfs, ignore_index=True)

    # Filter to double-sufficient cells
    soep_filtered, linkedin_filtered, _ = filter_to_double_sufficient_cells(
        soep_df=soep_all,
        linkedin_df=linkedin_all,
        cell_cols=STRATA_COLS,
        min_soep_obs=MIN_SOEP_OBS,
        min_linkedin_obs=MIN_LINKEDIN_OBS,
    )

    # Apply KNN-based selection
    linkedin_selected, metadata = knn_select_within_cells(
        linkedin_df=linkedin_filtered,
        soep_df=soep_filtered,
        cell_cols=STRATA_COLS,
        matches_per_soep=MATCHES_PER_SOEP,
        random_state=RANDOM_STATE,
    )

    # Add configuration to metadata
    metadata["config"] = {
        "min_soep_obs": MIN_SOEP_OBS,
        "min_linkedin_obs": MIN_LINKEDIN_OBS,
        "matches_per_soep": MATCHES_PER_SOEP,
        "random_state": RANDOM_STATE,
        "cell_cols": STRATA_COLS,
    }

    # Save selected LinkedIn data
    linkedin_selected.to_parquet(produces["data"], engine="fastparquet", index=False)

    # Save metadata
    with produces["metadata"].open("w") as f:
        json.dump(metadata, f, indent=2)
