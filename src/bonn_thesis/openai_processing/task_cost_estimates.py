"""Pytask for estimating costs of OpenAI batch processing and fine-tuning."""

import json
from pathlib import Path

import pandas as pd
import yaml

from bonn_thesis.config import OCCUPATION_DATA_BLD, SOEP_DATA_BLD, SRC
from bonn_thesis.openai_processing.isco_cost_estimates import estimate_fine_tune_costs
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


isco_dependencies = {
    "config": SRC
    / "openai_processing"
    / "configs"
    / "fine_tune"
    / "isco_classifier_03.yaml",
    "training_file": OCCUPATION_DATA_BLD
    / "openai_fine_tune"
    / "isco_training_data_03.jsonl",
    "validation_file": OCCUPATION_DATA_BLD
    / "openai_fine_tune"
    / "isco_validation_data_03.jsonl",
}


def task_estimate_costs_isco_fine_tune(
    depends_on=isco_dependencies,
    produces=OCCUPATION_DATA_BLD / "fine_tune_metadata.csv",
):
    """Estimate costs for ISCO fine-tuning and create metadata.

    This task:
    1. Loads fine-tuning configuration
    2. Counts tokens in training and validation files
    3. Estimates training and inference costs
    4. Creates metadata CSV with all information
    """
    # Load configuration
    with Path(depends_on["config"]).open() as f:
        config = yaml.safe_load(f)

    training_file = Path(depends_on["training_file"])
    validation_file = Path(depends_on["validation_file"])

    # Estimate costs
    cost_estimates = estimate_fine_tune_costs(
        training_jsonl_path=training_file,
        validation_jsonl_path=validation_file if validation_file.exists() else None,
        config=config,
        n_predictions_to_estimate=1000,
    )

    # Add additional metadata fields from config
    metadata = {
        **cost_estimates,
        "description": config["description"],
        "training_file": str(training_file),
        "validation_file": str(validation_file) if validation_file.exists() else None,
        "system_message": config["system_message"],
        "model_suffix": config.get("model_suffix", ""),
    }

    # Create metadata DataFrame and save
    df = pd.DataFrame([metadata])
    output_path = Path(produces)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
