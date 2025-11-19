"""Tasks for extracting LinkedIn experience data in batches."""

import pandas as pd
from pytask import task

from bonn_thesis.config import BATCH_SIZE, RAW_DATA_BLD
from bonn_thesis.data_management.sql_queries import extract_merged_linkedin_data

MAX_BATCHES_EXP = 563


for batch_num in range(1, MAX_BATCHES_EXP + 1):

    @task(id=f"batch_{batch_num:03d}", kwargs={"batch_num": batch_num})
    def task_extract_experience_linkedin_data(
        batch_num: int,
        produces=RAW_DATA_BLD / f"linkedin_experience_{batch_num:03d}.parquet",
    ):
        """Extract a single batch of LinkedIn experience data.

        Args:
            batch_num: Batch number to extract (1-indexed)
            produces: Output parquet file path
        """
        offset = (batch_num - 1) * BATCH_SIZE

        batch_data = extract_merged_linkedin_data(limit=BATCH_SIZE, offset=offset)

        if batch_data.empty:
            pd.DataFrame().to_parquet(produces, index=False)
            return

        batch_data.to_parquet(produces, index=False)
