"""Pytask for preparing SOEP data and JSONL files for OpenAI batch processing."""

from pathlib import Path

import pandas as pd
import yaml

from bonn_thesis.config import SOEP_DATA_BLD, SRC
from bonn_thesis.openai_processing.soep_agg_partition import (
    aggregate_and_partition_soep,
)
from bonn_thesis.openai_processing.soep_prepare_jsonl_files import (
    prepare_jsonl_for_openai,
    save_jsonl,
)

dependencies = {
    "soep_clean": SOEP_DATA_BLD / "soep_clean.parquet",
    "agg_config": SRC
    / "openai_processing"
    / "configs"
    / "data_agg"
    / "soep_agg_01.yaml",
    "partition_config": SRC
    / "openai_processing"
    / "configs"
    / "data_partition"
    / "soep_partition_01.yaml",
    "experiment_config": SRC
    / "openai_processing"
    / "configs"
    / "experiments"
    / "wage_soep_exp_01.yaml",
    "prompt_config": SRC
    / "openai_processing"
    / "configs"
    / "prompts"
    / "prompts_wage_soep_01.yaml",
}

products = {
    "jsonl": SOEP_DATA_BLD / "openai_inputs" / "wage_soep_exp_01.jsonl",
    "reference_data": SOEP_DATA_BLD / "aggregated" / "soep_agg_part_01.parquet",
}


def task_soep_prepare_data_for_openai(
    depends_on=dependencies,
    produces=products,
):
    df = pd.read_parquet(depends_on["soep_clean"], engine="fastparquet")

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
