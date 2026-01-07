"""Tasks to clean location data in LinkedIn experience files."""

from pathlib import Path

import pandas as pd
from pytask import task

from bonn_thesis.config import (
    BUNDESLAND_DATA_BLD,
    EXPERIENCE_DATA_BLD,
    LOCATION_DATA_BLD,
)
from bonn_thesis.data_management.clean_location_data import clean_location_data

_clean_files = list(EXPERIENCE_DATA_BLD.glob("linkedin_experience_clean_*.parquet"))

for clean_file in _clean_files:
    suffix = clean_file.stem.replace("linkedin_experience_clean_", "")

    @task(
        id=suffix,
        kwargs={
            "input_file": clean_file,
            "bundesland_data": BUNDESLAND_DATA_BLD / "bundesland_reference.parquet",
        },
    )
    def task_clean_location_data(
        input_file: Path,
        bundesland_data: Path,
        produces=LOCATION_DATA_BLD
        / f"linkedin_experience_with_location_{suffix}.parquet",
    ) -> None:
        """Clean location data for a single LinkedIn experience file.

        Args:
            input_file: Path to the cleaned LinkedIn experience file.
            bundesland_data: Path to the bundesland reference data.
            produces: Path to save the file with matched locations.
        """
        experience_data = pd.read_parquet(input_file)
        bundesland_reference = pd.read_parquet(bundesland_data)

        produces.parent.mkdir(parents=True, exist_ok=True)

        if len(experience_data) == 0:
            pd.DataFrame().to_parquet(produces, index=False)
            return

        location_data = clean_location_data(experience_data, bundesland_reference)
        location_data.to_parquet(produces, index=False)
