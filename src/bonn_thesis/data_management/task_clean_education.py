"""Tasks to clean LinkedIn education data."""

from collections import Counter
from pathlib import Path

import pandas as pd
from pytask import task

from bonn_thesis.config import EDUCATION_DATA_BLD, RAW_DATA_BLD
from bonn_thesis.data_management.clean_education_data import clean_education_data
from bonn_thesis.data_management.education_identify_degree import (
    aggregate_unclassified_degrees,
    get_top_n_degrees,
)


def task_extract_top_degree_types(
    produces=EDUCATION_DATA_BLD / "top_300_degree_types.csv",
):
    """Extract top 300 unclassified degree types and save to CSV."""
    education_files = sorted(RAW_DATA_BLD.glob("linkedin_education_*.parquet"))

    degree_counter = Counter()

    for file_path in education_files:
        df = pd.read_parquet(
            file_path,
            columns=["degree_type", "start_date", "end_date", "case_degree_label"],
            engine="fastparquet",
        )

        file_counter = aggregate_unclassified_degrees(df)
        degree_counter.update(file_counter)

        del df

    top_300_df = get_top_n_degrees(degree_counter, n=300)
    top_300_df.to_csv(produces, index=False)


_raw_files = list(RAW_DATA_BLD.glob("linkedin_education_*.parquet"))

for raw_file in _raw_files:
    suffix = raw_file.stem.replace("linkedin_education_", "")

    @task(id=suffix, kwargs={"input_file": raw_file})
    def task_clean_linkedin_education_data(
        input_file: Path,
        depends_on: Path = EDUCATION_DATA_BLD / "top_degree_types_classified.csv",
        produces=EDUCATION_DATA_BLD / f"linkedin_education_clean_{suffix}.parquet",
    ) -> None:
        """Clean a single LinkedIn education data file.

        Args:
            input_file: Path to the raw LinkedIn education file.
            depends_on: Path to the degree type classification mapping CSV.
            produces: Path to save the cleaned education file.
        """
        if depends_on.exists():
            degree_mapping = pd.read_csv(depends_on)
            mapping_dict = degree_mapping.set_index("degree_type_cleaned")[
                "education_level"
            ].to_dict()
        else:
            mapping_dict = {}

        raw_data = pd.read_parquet(input_file)

        if len(raw_data) == 0:
            pd.DataFrame().to_parquet(produces, index=False)
            return

        clean_data = clean_education_data(raw_data, mapping_dict)

        produces.parent.mkdir(parents=True, exist_ok=True)
        clean_data.to_parquet(produces, index=False)
