"""Tasks for applying sample selection to experience data files."""

import json
from pathlib import Path

import pandas as pd
from pytask import task

from bonn_thesis.config import (
    EDUCATION_DATA_BLD,
    LOCATION_DATA_BLD,
    SAMPLE_SELECTION_BLD,
)
from bonn_thesis.data_management.sample_selection import run_sample_selection

_experience_files = list(
    LOCATION_DATA_BLD.glob("linkedin_experience_with_location_*.parquet")
)

for exp_file in _experience_files:
    # Extract batch number from experience file
    suffix = exp_file.stem.replace("linkedin_experience_with_location_", "")

    # Match with corresponding education file
    edu_file = EDUCATION_DATA_BLD / f"linkedin_education_clean_{suffix}.parquet"

    # Create produces dict for this specific batch
    batch_produces = {
        "data": SAMPLE_SELECTION_BLD / f"linkedin_experience_selected_{suffix}.parquet",
        "tracking": SAMPLE_SELECTION_BLD / f"sample_selection_tracking_{suffix}.csv",
        "metadata": SAMPLE_SELECTION_BLD / f"sample_selection_metadata_{suffix}.json",
    }

    @task(
        id=suffix,
        kwargs={
            "exp_file": exp_file,
            "edu_file": edu_file,
            "produces": batch_produces,
        },
    )
    def task_sample_selection(
        exp_file: Path,
        edu_file: Path,
        produces: dict,
    ) -> None:
        """Apply sample selection to a single batch of experience and education data.

        Args:
            exp_file: Path to the experience data file with location information.
            edu_file: Path to the education data file.
            produces: Dictionary with paths for data, tracking, and metadata outputs.
        """
        # Load data
        experience_data = pd.read_parquet(exp_file, engine="fastparquet")
        education_data = pd.read_parquet(edu_file, engine="fastparquet")

        # Handle empty dataframes
        produces["data"].parent.mkdir(parents=True, exist_ok=True)

        if len(experience_data) == 0 or len(education_data) == 0:
            pd.DataFrame().to_parquet(
                produces["data"], engine="fastparquet", index=False
            )
            pd.DataFrame().to_csv(produces["tracking"], index=False)

            metadata = {
                "source_experience_file": exp_file.name,
                "source_education_file": edu_file.name,
                "output_file": produces["data"].name,
                "processing_date": pd.Timestamp.now().isoformat(),
                "initial_rows": 0,
                "final_rows": 0,
                "total_removed": 0,
                "removal_rate": 0.0,
                "note": "Empty input data",
            }
            with produces["metadata"].open("w") as f:
                json.dump(metadata, f, indent=2, default=str)
            return

        # Run sample selection
        final_data, tracking_log = run_sample_selection(experience_data, education_data)

        # Save output data
        final_data.to_parquet(produces["data"], engine="fastparquet", index=False)

        # Save tracking log
        tracking_df = pd.DataFrame(tracking_log)
        tracking_df["experience_file"] = exp_file.name
        tracking_df["education_file"] = edu_file.name
        tracking_df["timestamp"] = pd.Timestamp.now()
        tracking_df.to_csv(produces["tracking"], index=False)

        # Save metadata
        metadata = {
            "source_experience_file": exp_file.name,
            "source_education_file": edu_file.name,
            "output_file": produces["data"].name,
            "processing_date": pd.Timestamp.now().isoformat(),
            "initial_rows": tracking_log[0]["rows_after"] if tracking_log else 0,
            "final_rows": tracking_log[-1]["rows_after"] if tracking_log else 0,
            "total_removed": (
                tracking_log[0]["rows_after"] - tracking_log[-1]["rows_after"]
                if tracking_log
                else 0
            ),
            "removal_rate": (
                (tracking_log[0]["rows_after"] - tracking_log[-1]["rows_after"])
                / tracking_log[0]["rows_after"]
                if tracking_log and tracking_log[0]["rows_after"] > 0
                else 0
            ),
        }

        with produces["metadata"].open("w") as f:
            json.dump(metadata, f, indent=2, default=str)
