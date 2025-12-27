"""Functions for estimating costs of OpenAI batch processing and managing metadata."""

from datetime import UTC, datetime

import pandas as pd
import tiktoken

from bonn_thesis.config import (
    BATCH_API_DISCOUNT,
    DEFAULT_OPENAI_MODEL,
    OPENAI_API_MODELS,
)


def count_tokens(requests, encoding_name="cl100k_base"):
    """Count tokens from list of request dictionaries."""
    encoding = tiktoken.get_encoding(encoding_name)

    total_input_tokens = 0
    total_estimated_output_tokens = 0
    n_requests = 0

    for request in requests:
        body = request.get("body", {})
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 100)

        # Count tokens in messages (system + user + overhead)
        for message in messages:
            content_tokens = len(encoding.encode(message.get("content", "")))
            total_input_tokens += content_tokens + 4  # 4 tokens overhead per message

        total_estimated_output_tokens += max_tokens
        n_requests += 1

    return {
        "n_requests": n_requests,
        "total_input_tokens": total_input_tokens,
        "total_estimated_output_tokens": total_estimated_output_tokens,
        "avg_input_tokens_per_request": total_input_tokens / n_requests
        if n_requests > 0
        else 0,
    }


def calculate_costs(token_counts, model_pricing, *, apply_batch_discount=True):
    """Calculate estimated costs based on token counts and pricing."""
    input_cost = (token_counts["total_input_tokens"] / 1_000_000) * model_pricing[
        "input"
    ]
    output_cost = (
        token_counts["total_estimated_output_tokens"] / 1_000_000
    ) * model_pricing["output"]

    if apply_batch_discount:
        input_cost *= BATCH_API_DISCOUNT
        output_cost *= BATCH_API_DISCOUNT

    return {
        "estimated_input_cost": round(input_cost, 4),
        "estimated_output_cost": round(output_cost, 4),
        "estimated_total_cost": round(input_cost + output_cost, 4),
    }


def extract_metadata(experiment_config, jsonl_path, reference_data_path):
    """Extract metadata from loaded experiment config."""
    return {
        "batch_name": experiment_config.get("batch_name", ""),
        "experiment_id": experiment_config.get("experiment_id", ""),
        "description": experiment_config.get("description", ""),
        "model": experiment_config.get("model", ""),
        "temperature": experiment_config.get("temperature", 0.7),
        "max_tokens": experiment_config.get("max_tokens", 100),
        "jsonl_input_path": str(jsonl_path),
        "reference_data_path": str(reference_data_path),
    }


def create_metadata_row(metadata, token_counts, costs):
    """Create metadata row dictionary."""
    return {
        "batch_id": None,
        "batch_name": metadata["batch_name"],
        "experiment_id": metadata["experiment_id"],
        "description": metadata["description"],
        "model": metadata["model"],
        "temperature": metadata["temperature"],
        "max_tokens": metadata["max_tokens"],
        "n_requests": token_counts["n_requests"],
        "estimated_input_tokens": token_counts["total_input_tokens"],
        "estimated_output_tokens": token_counts["total_estimated_output_tokens"],
        "avg_input_tokens_per_request": round(
            token_counts["avg_input_tokens_per_request"], 2
        ),
        "estimated_input_cost": costs["estimated_input_cost"],
        "estimated_output_cost": costs["estimated_output_cost"],
        "estimated_total_cost": costs["estimated_total_cost"],
        "actual_input_tokens": None,
        "actual_output_tokens": None,
        "actual_total_cost": None,
        "status": "prepared",
        "created_at": datetime.now(UTC).isoformat(),
        "submitted_at": None,
        "completed_at": None,
        "jsonl_input_path": metadata["jsonl_input_path"],
        "jsonl_output_path": None,
        "reference_data_path": metadata["reference_data_path"],
    }


def update_metadata_df(existing_df, row_data):
    """Update or append row to metadata DataFrame."""
    if existing_df.empty or len(existing_df.columns) == 0:
        return pd.DataFrame([row_data])

    # Update existing or append new
    existing_idx = existing_df[
        existing_df["batch_name"] == row_data["batch_name"]
    ].index
    if len(existing_idx) > 0:
        for key, value in row_data.items():
            existing_df.loc[existing_idx[0], key] = value
        return existing_df
    return pd.concat([existing_df, pd.DataFrame([row_data])], ignore_index=True)


def estimate_batch_costs(
    requests,
    experiment_config,
    jsonl_path,
    reference_data_path,
    existing_metadata_df,
    *,
    apply_batch_discount=True,
):
    """Estimate costs and create metadata for OpenAI batch processing.

    Args:
        requests: List of request dictionaries from JSONL
        experiment_config: Loaded experiment configuration dict
        jsonl_path: Path to JSONL file (for metadata only)
        reference_data_path: Path to reference data file (for metadata only)
        existing_metadata_df: Existing metadata DataFrame
        apply_batch_discount: Whether to apply batch API discount

    Returns:
        pd.DataFrame: Updated metadata DataFrame

    Raises:
        ValueError: If model specified in config is not found in OPENAI_API_MODELS
    """
    model_name = experiment_config.get("model", DEFAULT_OPENAI_MODEL)

    # Get pricing from config
    if model_name not in OPENAI_API_MODELS:
        available_models = ", ".join(OPENAI_API_MODELS.keys())
        msg = (
            f"Model '{model_name}' is not configured in config.py.\n"
            f"Available models: {available_models}\n"
            f"Please add pricing for '{model_name}' to OPENAI_API_MODELS in config.py"
        )
        raise ValueError(msg)

    model_config = OPENAI_API_MODELS[model_name]
    model_pricing = {
        "input": model_config["input_cost"],
        "output": model_config["output_cost"],
    }

    # Calculate costs
    token_counts = count_tokens(requests)
    costs = calculate_costs(
        token_counts, model_pricing, apply_batch_discount=apply_batch_discount
    )
    metadata = extract_metadata(experiment_config, jsonl_path, reference_data_path)
    row_data = create_metadata_row(metadata, token_counts, costs)
    updated_df = update_metadata_df(existing_metadata_df, row_data)

    return updated_df


def update_batch_status(
    metadata_csv_path,
    batch_name,
    status=None,
    batch_id=None,
    submitted_at=None,
    completed_at=None,
    actual_tokens=None,
    actual_cost=None,
    output_path=None,
):
    """Update batch status in metadata CSV after submission or completion."""
    df = pd.read_csv(metadata_csv_path)

    idx = df[df["batch_name"] == batch_name].index
    if len(idx) == 0:
        msg = f"Batch '{batch_name}' not found in metadata CSV"
        raise ValueError(msg)

    idx = idx[0]

    # Update fields
    if status:
        df.loc[idx, "status"] = status
    if batch_id:
        df.loc[idx, "batch_id"] = batch_id
    if submitted_at:
        df.loc[idx, "submitted_at"] = submitted_at
    if completed_at:
        df.loc[idx, "completed_at"] = completed_at
    if actual_tokens:
        df.loc[idx, "actual_input_tokens"] = actual_tokens.get("input")
        df.loc[idx, "actual_output_tokens"] = actual_tokens.get("output")
    if actual_cost:
        df.loc[idx, "actual_total_cost"] = actual_cost
    if output_path:
        df.loc[idx, "jsonl_output_path"] = str(output_path)

    df.to_csv(metadata_csv_path, index=False)

    return df
