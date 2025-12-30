"""Pytask for estimating costs of OpenAI batch processing."""

import json
from pathlib import Path

import pandas as pd
import yaml

from bonn_thesis.config import SOEP_DATA_BLD, SRC
from bonn_thesis.openai_processing.soep_cost_estimates import estimate_batch_costs

dependencies = {
    "jsonl": SOEP_DATA_BLD / "openai_inputs" / "wage_soep_exp_16.jsonl",
    "experiment_config": SRC
    / "openai_processing"
    / "configs"
    / "experiments"
    / "wage_soep_exp_16.yaml",
    "reference_data": SOEP_DATA_BLD / "aggregated" / "soep_agg_part_16.parquet",
}


def task_estimate_costs_wage_soep_exp(
    depends_on=dependencies,
    produces=SOEP_DATA_BLD / "soep_metadata.csv",
):
    """Estimate costs for OpenAI batch processing and save metadata.

    This task handles all file I/O:
    1. Loads JSONL requests
    2. Loads experiment configuration
    3. Loads or creates metadata DataFrame
    4. Calls estimation functions
    5. Saves updated metadata to CSV
    """
    # Load JSONL requests
    with Path(depends_on["jsonl"]).open() as f:
        requests = [json.loads(line) for line in f]

    # Load experiment config
    with Path(depends_on["experiment_config"]).open() as f:
        experiment_config = yaml.safe_load(f)

    # Load or create metadata DataFrame
    metadata_path = Path(produces)
    if metadata_path.exists():
        try:
            existing_df = pd.read_csv(metadata_path)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            existing_df = pd.DataFrame()
    else:
        existing_df = pd.DataFrame()

    # Estimate costs and update metadata
    updated_df = estimate_batch_costs(
        requests=requests,
        experiment_config=experiment_config,
        jsonl_path=depends_on["jsonl"],
        reference_data_path=depends_on["reference_data"],
        existing_metadata_df=existing_df,
    )

    # Save updated metadata
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    updated_df.to_csv(metadata_path, index=False)
