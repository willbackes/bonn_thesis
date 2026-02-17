"""Tasks for assigning SOEP strata identifiers to LinkedIn data."""

import pickle
from pathlib import Path

import pandas as pd
from pytask import task

from bonn_thesis.config import (
    MERGED_EXP_ED_BLD,
    MERGED_EXP_ED_SAMPLING_BLD,
    SOEP_DATA_BLD,
)
from bonn_thesis.sampling.define_strata import create_cell_lookup
from bonn_thesis.sampling.identify_strata import assign_custom_id_to_linkedin


@task(
    id="create_cell_lookup",
    kwargs={
        "produces": MERGED_EXP_ED_SAMPLING_BLD / "cell_lookup.pkl",
    },
)
def task_create_cell_lookup(produces: Path) -> None:
    """Create SOEP cell lookup dictionary and save it.

    Args:
        produces: Path to save the cell lookup dictionary.
    """
    soep_agg_file = SOEP_DATA_BLD / "aggregated" / "soep_agg_part_15.parquet"
    soep_cells = pd.read_parquet(soep_agg_file)

    cell_lookup = create_cell_lookup(soep_cells)

    with produces.open("wb") as f:
        pickle.dump(cell_lookup, f)


# Path to cell lookup file (dependency for all LinkedIn tasks)
cell_lookup_file = MERGED_EXP_ED_SAMPLING_BLD / "cell_lookup.pkl"

# Find all LinkedIn merged files
_linkedin_files = list(MERGED_EXP_ED_BLD.glob("linkedin_merged_exp_ed_*.parquet"))

for linkedin_file in _linkedin_files:
    # Extract batch number from file
    suffix = linkedin_file.stem.replace("linkedin_merged_exp_ed_", "")

    # Create produces dict for this specific batch
    batch_produces = {
        "data": MERGED_EXP_ED_SAMPLING_BLD
        / f"linkedin_merged_exp_ed_strata_{suffix}.parquet",
    }

    @task(
        id=suffix,
        kwargs={
            "linkedin_file": linkedin_file,
            "cell_lookup_file": cell_lookup_file,
            "produces": batch_produces,
        },
    )
    def task_assign_strata(
        linkedin_file: Path,
        cell_lookup_file: Path,
        produces: dict,
    ) -> None:
        """Assign SOEP strata identifiers (custom_id and n_obs) to LinkedIn data.

        Args:
            linkedin_file: Path to LinkedIn merged data parquet file.
            cell_lookup_file: Path to cell lookup dictionary pickle file.
            produces: Dictionary with paths for data output.
        """
        # Ensure output directory exists
        produces["data"].parent.mkdir(parents=True, exist_ok=True)

        # Handle missing or empty input files
        if not linkedin_file.exists():
            pd.DataFrame().to_parquet(
                produces["data"], engine="fastparquet", index=False
            )
            return

        # Load cell lookup dictionary
        with cell_lookup_file.open("rb") as f:
            cell_lookup = pickle.load(f)

        # Load LinkedIn data
        linkedin_df = pd.read_parquet(linkedin_file, engine="fastparquet")

        if len(linkedin_df) == 0:
            pd.DataFrame().to_parquet(
                produces["data"], engine="fastparquet", index=False
            )
            return

        # Assign custom_id and soep_n_obs
        linkedin_with_strata = assign_custom_id_to_linkedin(linkedin_df, cell_lookup)

        # Save output data
        linkedin_with_strata.to_parquet(
            produces["data"], engine="fastparquet", index=False
        )
