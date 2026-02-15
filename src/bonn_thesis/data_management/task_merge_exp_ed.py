"""Tasks for merging experience, education, and ISCO classification data."""

from pathlib import Path

import pandas as pd
from pytask import task

from bonn_thesis.config import (
    BUNDESLAND_DATA_BLD,
    EDUCATION_DATA_BLD,
    MERGED_EXP_ED_BLD,
    OCCUPATION_DATA_BLD,
    RAW_DATA,
    SAMPLE_SELECTION_BLD,
)
from bonn_thesis.data_management.linkedin_merge_exp_ed_final import (
    merge_exp_ed_pipeline,
)

# Load reference data once (shared across all tasks)
isco_reference_file = RAW_DATA / "ISCO-08 EN Structure and definitions.xlsx"
bundesland_reference_file = BUNDESLAND_DATA_BLD / "bundesland_reference.parquet"

# Load reference data
isco_reference_df = pd.read_excel(isco_reference_file)
bundesland_reference_df = pd.read_parquet(
    bundesland_reference_file, engine="fastparquet"
)

# Find all sample selection files
_sample_files = list(
    SAMPLE_SELECTION_BLD.glob("linkedin_experience_selected_*.parquet")
)

for sample_file in _sample_files:
    # Extract batch number from sample file
    suffix = sample_file.stem.replace("linkedin_experience_selected_", "")

    # Match with corresponding ISCO and education files
    isco_file = (
        OCCUPATION_DATA_BLD
        / "openai_responses"
        / f"isco_classification_{suffix}_results.jsonl"
    )
    edu_file = EDUCATION_DATA_BLD / f"linkedin_education_clean_{suffix}.parquet"

    # Create produces dict for this specific batch
    batch_produces = {
        "data": MERGED_EXP_ED_BLD / f"linkedin_merged_exp_ed_{suffix}.parquet",
    }

    @task(
        id=suffix,
        kwargs={
            "sample_file": sample_file,
            "isco_file": isco_file,
            "edu_file": edu_file,
            "isco_reference_df": isco_reference_df,
            "bundesland_reference_df": bundesland_reference_df,
            "produces": batch_produces,
        },
    )
    def task_merge_exp_ed(
        sample_file: Path,
        isco_file: Path,
        edu_file: Path,
        isco_reference_df: pd.DataFrame,
        bundesland_reference_df: pd.DataFrame,
        produces: dict,
    ) -> None:
        """Merge experience, education, and ISCO data for a single batch.

        Args:
            sample_file: Path to the sample selection parquet file.
            isco_file: Path to the ISCO classification JSONL file.
            edu_file: Path to the education data parquet file.
            isco_reference_df: DataFrame with ISCO reference data.
            bundesland_reference_df: DataFrame with bundesland reference data.
            produces: Dictionary with paths for data output.
        """
        # Ensure output directory exists
        produces["data"].parent.mkdir(parents=True, exist_ok=True)

        # Handle missing or empty input files
        if not isco_file.exists() or not sample_file.exists() or not edu_file.exists():
            pd.DataFrame().to_parquet(
                produces["data"], engine="fastparquet", index=False
            )
            return

        sample_df = pd.read_parquet(sample_file, engine="fastparquet")
        education_df = pd.read_parquet(edu_file, engine="fastparquet")

        if len(sample_df) == 0 or len(education_df) == 0:
            pd.DataFrame().to_parquet(
                produces["data"], engine="fastparquet", index=False
            )
            return

        # Run merge pipeline
        final_data = merge_exp_ed_pipeline(
            isco_file=isco_file,
            sample_file=sample_file,
            education_file=edu_file,
            isco_reference_df=isco_reference_df,
            bundesland_reference_df=bundesland_reference_df,
        )

        # Save output data
        final_data.to_parquet(produces["data"], engine="fastparquet", index=False)
