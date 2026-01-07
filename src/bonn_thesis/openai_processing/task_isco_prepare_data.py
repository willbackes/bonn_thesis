"""Pytask for preparing ISCO fine-tuning data and JSONL files."""

from pathlib import Path

import pandas as pd
import yaml

from bonn_thesis.config import OCCUPATION_DATA_BLD, RAW_DATA, SRC
from bonn_thesis.openai_processing.isco_prepare_jsonl_files import (
    create_fine_tune_jsonl,
    split_train_validation,
)

# Configuration file
FINE_TUNE_CONFIG = (
    SRC / "openai_processing" / "configs" / "fine_tune" / "isco_classifier_03.yaml"
)

# Data file
ISCO_DATA_FILE = RAW_DATA / "isco_esco_occupations_taxonomy.xlsx"

dependencies = {
    "config": FINE_TUNE_CONFIG,
    "data": ISCO_DATA_FILE,
}

products = {
    "training_file": OCCUPATION_DATA_BLD
    / "openai_fine_tune"
    / "isco_training_data_03.jsonl",
    "validation_file": OCCUPATION_DATA_BLD
    / "openai_fine_tune"
    / "isco_validation_data_03.jsonl",
}


def task_isco_prepare_fine_tune_data(
    depends_on=dependencies,
    produces=products,
):
    """Prepare ISCO classification data for OpenAI fine-tuning.

    Args:
        depends_on: Dictionary with paths to config file and data
        produces: Dictionary with paths to output files
    """
    # Load configuration
    with Path(depends_on["config"]).open() as f:
        config = yaml.safe_load(f)

    # Load data
    data = pd.read_excel(depends_on["data"])

    # Create ISCO_3_DIGIT from ISCO_4_DIGIT
    data["ISCO_3_DIGIT"] = data["ISCO_4_DIGIT"].apply(
        lambda x: "NA" if pd.isna(x) else str(x).zfill(4)[:3]
    )

    # Get columns and clean data
    input_col = config["input_column"]
    output_col = config["output_column"]
    data_clean = data[[input_col, output_col]].dropna().drop_duplicates()

    # Split into train/validation if configured
    if config.get("train_test_split"):
        train_data, val_data = split_train_validation(
            data_clean,
            split_ratio=config["train_test_split"],
            random_seed=config.get("random_seed", 42),
        )
    else:
        train_data = data_clean
        val_data = None

    # Create output directory
    produces["training_file"].parent.mkdir(parents=True, exist_ok=True)

    # Create training JSONL
    create_fine_tune_jsonl(
        data=train_data,
        input_col=input_col,
        output_col=output_col,
        system_message=config["system_message"],
        output_path=produces["training_file"],
    )

    # Create validation JSONL if split was used
    if val_data is not None:
        create_fine_tune_jsonl(
            data=val_data,
            input_col=input_col,
            output_col=output_col,
            system_message=config["system_message"],
            output_path=produces["validation_file"],
        )
