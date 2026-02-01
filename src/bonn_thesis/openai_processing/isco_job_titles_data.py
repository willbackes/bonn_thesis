"""Functions for preparing ISCO classification JSONL files for OpenAI batch API."""

import json

import pandas as pd


def create_classification_request(row, experiment_config, batch_number):
    """Create a single JSONL request for ISCO classification.

    Args:
        row: DataFrame row with exp_id, job_title, and industry
        experiment_config: Experiment configuration (model, temperature, etc.)
        batch_number: Batch number for unique ID generation

    Returns:
        dict: Request object for JSONL
    """
    batch_name = f"{experiment_config['batch_name_prefix']}_{batch_number:03d}"
    custom_id = f"{batch_name}_{row['exp_id']}"

    # Format user content as "job_title. Industry: industry"
    # or just "job_title." if industry is empty
    if (
        pd.isna(row["industry"])
        or row["industry"] == "None"
        or str(row["industry"]).strip() == ""
    ):
        user_content = f"{row['job_title']}."
    else:
        user_content = f"{row['job_title']}. Industry: {row['industry']}"

    request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": experiment_config["model"],
            "messages": [
                {
                    "role": "system",
                    "content": experiment_config.get(
                        "system_message",
                        (
                            "You are a job classifier. Given a job title, "
                            "classify the job according to the ISCO-08 "
                            "taxonomy at the 3-digit level."
                        ),
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            "temperature": experiment_config.get("temperature", 0),
            "max_tokens": experiment_config.get("max_tokens", 100),
        },
    }

    return request


def save_classification_jsonl(requests, output_path):
    """Save classification requests to JSONL file.

    Args:
        requests: List of request dictionaries
        output_path: Path to output JSONL file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")


def prepare_classification_batch(df, experiment_config, batch_number):
    """Prepare JSONL requests for a batch of experiences.

    Args:
        df: DataFrame with experience data (must have 'exp_id', 'job_title', 'industry')
        experiment_config: Experiment configuration
        batch_number: Batch number for tracking

    Returns:
        tuple: (list of requests, reference DataFrame)
    """
    batch_name = f"{experiment_config['batch_name_prefix']}_{batch_number:03d}"

    # Validate required columns
    required_cols = ["exp_id", "job_title", "industry"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        msg = f"Missing required columns: {missing_cols}"
        raise ValueError(msg)

    # Create requests
    requests = []
    for _idx, row in df.iterrows():
        request = create_classification_request(row, experiment_config, batch_number)
        requests.append(request)

    # Create reference data for joining results back
    reference_df = df[["exp_id", "job_title", "industry"]].copy()
    reference_df["custom_id"] = [
        f"{batch_name}_{exp_id}" for exp_id in reference_df["exp_id"]
    ]
    reference_df["batch_number"] = batch_number
    reference_df["batch_name"] = batch_name

    return requests, reference_df
