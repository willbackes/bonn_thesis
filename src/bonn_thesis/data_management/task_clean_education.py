from collections import Counter

import pandas as pd

from bonn_thesis.config import EDUCATION_DATA_BLD, RAW_DATA_BLD
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
