"""Tasks to clean LinkedIn experience data."""

from pathlib import Path

import pandas as pd
from pytask import task

from bonn_thesis.config import EXPERIENCE_DATA_BLD, RAW_DATA_BLD
from bonn_thesis.data_management.clean_experience_data import clean_experience_data

_raw_files = list(RAW_DATA_BLD.glob("linkedin_experience_*.parquet"))

for raw_file in _raw_files:
    suffix = raw_file.stem.replace("linkedin_experience_", "")

    @task(id=suffix, kwargs={"input_file": raw_file})
    def task_clean_linkedin_experience_data(
        input_file: Path,
        produces=EXPERIENCE_DATA_BLD / f"linkedin_experience_clean_{suffix}.parquet",
    ) -> None:
        """Clean a single LinkedIn experience data file.

        Args:
            input_file: Path to the raw LinkedIn data file.
            produces: Path to save the cleaned data file.
        """
        raw_data = pd.read_parquet(input_file)
        if len(raw_data) == 0:
            pd.DataFrame().to_parquet(produces, index=False)
            return
        clean_data = clean_experience_data(raw_data)
        produces.parent.mkdir(parents=True, exist_ok=True)
        clean_data.to_parquet(produces, index=False)
