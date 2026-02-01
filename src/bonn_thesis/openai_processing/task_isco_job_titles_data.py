"""Task to prepare JSONL files for ISCO classification."""

import pandas as pd
import yaml
from pytask import task

from bonn_thesis.config import (
    OCCUPATION_DATA_BLD,
    SAMPLE_SELECTION_BLD,
    SRC,
)
from bonn_thesis.openai_processing.isco_job_titles_data import (
    prepare_classification_batch,
    save_classification_jsonl,
)

# Find all sample files
sample_files = sorted(
    SAMPLE_SELECTION_BLD.glob("linkedin_experience_selected_*.parquet")
)

# Configuration file path
config_path = (
    SRC
    / "openai_processing"
    / "configs"
    / "occupation_classification"
    / "occupation_classification_001.yaml"
)


for sample_file in sample_files:
    # Extract batch number from filename
    # (e.g., linkedin_experience_selected_001.parquet -> 001)
    batch_num_str = sample_file.stem.split("_")[-1]
    batch_num = int(batch_num_str)

    # Define output paths
    jsonl_output = (
        OCCUPATION_DATA_BLD
        / "openai_inputs"
        / f"isco_classification_{batch_num_str}.jsonl"
    )

    @task(id=f"prepare_isco_jsonl_{batch_num_str}")
    def task_prepare_isco_jsonl(
        sample_file=sample_file,
        config=config_path,
        produces=jsonl_output,
        batch_number=batch_num,
    ):
        """Prepare JSONL file for a single batch of LinkedIn experience data."""
        # Load config
        with config.open() as f:
            experiment_config = yaml.safe_load(f)

        # Load experience data
        df = pd.read_parquet(sample_file, engine="fastparquet")

        # Handle empty dataframes
        if len(df) == 0:
            # Create empty JSONL file
            produces.parent.mkdir(parents=True, exist_ok=True)
            produces.touch()
            return

        # Prepare batch
        requests, _ = prepare_classification_batch(
            df=df,
            experiment_config=experiment_config,
            batch_number=batch_number,
        )

        # Save JSONL
        save_classification_jsonl(requests, produces)
