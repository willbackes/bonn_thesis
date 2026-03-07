"""Pytask for preparing LinkedIn data and JSONL files for OpenAI batch processing."""

from pathlib import Path

import pandas as pd
import yaml

from bonn_thesis.config import LINKEDIN_MATCHED_TO_SOEP_BLD, SRC
from bonn_thesis.openai_processing.soep_agg_partition import (
    aggregate_and_partition_soep,
)
from bonn_thesis.openai_processing.soep_prepare_jsonl_files import (
    prepare_jsonl_for_openai,
    save_jsonl,
)

dependencies = {
    "linkedin_selected": LINKEDIN_MATCHED_TO_SOEP_BLD / "linkedin_selected.parquet",
    "agg_config": SRC
    / "openai_processing"
    / "configs"
    / "data_agg"
    / "linkedin_agg_14.yaml",
    "partition_config": SRC
    / "openai_processing"
    / "configs"
    / "data_partition"
    / "linkedin_partition_14.yaml",
    "experiment_config": SRC
    / "openai_processing"
    / "configs"
    / "experiments"
    / "wage_linkedin_exp_14.yaml",
    "prompt_config": SRC
    / "openai_processing"
    / "configs"
    / "prompts"
    / "prompts_wage_linkedin_14.yaml",
}

products = {
    "jsonl": LINKEDIN_MATCHED_TO_SOEP_BLD
    / "openai_inputs"
    / "wage_linkedin_exp_14.jsonl",
    "reference_data": LINKEDIN_MATCHED_TO_SOEP_BLD
    / "aggregated"
    / "linkedin_agg_part_14.parquet",
}


def task_linkedin_prepare_data(
    depends_on=dependencies,
    produces=products,
):
    """Prepare LinkedIn data for OpenAI batch processing.

    This task:
    1. Loads the selected LinkedIn data
    2. Applies aggregation and partitioning (individual level for exp 14)
    3. Creates JSONL requests for OpenAI batch API
    4. Saves reference data with custom_ids for matching results

    Args:
        depends_on: Dictionary of input file paths
        produces: Dictionary of output file paths
    """
    df = pd.read_parquet(depends_on["linkedin_selected"], engine="fastparquet")

    with Path(depends_on["agg_config"]).open() as f:
        agg_config = yaml.safe_load(f)

    with Path(depends_on["partition_config"]).open() as f:
        partition_config = yaml.safe_load(f)

    with Path(depends_on["experiment_config"]).open() as f:
        experiment_config = yaml.safe_load(f)

    with Path(depends_on["prompt_config"]).open() as f:
        prompt_config = yaml.safe_load(f)

    aggregated_df = aggregate_and_partition_soep(
        df=df,
        agg_config=agg_config,
        partition_config=partition_config,
    )

    requests, reference_df = prepare_jsonl_for_openai(
        aggregated_df=aggregated_df,
        experiment_config=experiment_config,
        prompt_config=prompt_config,
    )

    batch_name = experiment_config["batch_name"]
    reference_df["custom_id"] = [f"{batch_name}_{idx}" for idx in reference_df.index]

    save_jsonl(requests, produces["jsonl"])

    produces["reference_data"].parent.mkdir(parents=True, exist_ok=True)
    reference_df.to_parquet(
        produces["reference_data"], engine="fastparquet", index=False
    )
