"""Functions for managing ISCO classification batch API submissions and monitoring."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI

from bonn_thesis.config import BATCH_API_DISCOUNT, OPENAI_API_MODELS


def get_api_key():
    """Get OpenAI API key from environment variable."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        msg = (
            "OPENAI_API_KEY not found in environment variables. "
            "Set it with: export OPENAI_API_KEY='your-key-here'"
        )
        raise ValueError(msg)
    return api_key


def upload_batch_file(jsonl_path, api_key=None):
    """Upload JSONL file to OpenAI for batch processing."""
    if api_key is None:
        api_key = get_api_key()

    client = OpenAI(api_key=api_key)

    with Path(jsonl_path).open("rb") as f:
        file_response = client.files.create(file=f, purpose="batch")

    return file_response.id


def create_batch(file_id, experiment_config, batch_name, api_key=None):
    """Create a batch job with uploaded file.

    Args:
        file_id: OpenAI file ID from upload
        experiment_config: Experiment configuration dict
        batch_name: Name for this specific batch (e.g., isco_classification_001)
        api_key: OpenAI API key (if None, gets from environment)

    Returns:
        dict: Batch information (batch_id, batch_name, status, submitted_at)
    """
    if api_key is None:
        api_key = get_api_key()

    client = OpenAI(api_key=api_key)

    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "batch_name": batch_name,
            "experiment_id": experiment_config.get("experiment_id", ""),
            "description": experiment_config.get("description", ""),
        },
    )

    return {
        "batch_id": batch_response.id,
        "batch_name": batch_name,
        "status": batch_response.status,
        "submitted_at": datetime.now(UTC).isoformat(),
    }


def check_batch_status(batch_id, api_key=None):
    """Check the status of a batch job."""
    if api_key is None:
        api_key = get_api_key()

    client = OpenAI(api_key=api_key)
    batch = client.batches.retrieve(batch_id)

    return {
        "batch_id": batch.id,
        "status": batch.status,
        "created_at": batch.created_at,
        "in_progress_at": getattr(batch, "in_progress_at", None),
        "completed_at": getattr(batch, "completed_at", None),
        "failed_at": getattr(batch, "failed_at", None),
        "expired_at": getattr(batch, "expired_at", None),
        "request_counts": {
            "total": batch.request_counts.total,
            "completed": batch.request_counts.completed,
            "failed": batch.request_counts.failed,
        },
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
        "metadata": batch.metadata,
    }


def download_batch_results(batch_id, output_dir, api_key=None):
    """Download results from a completed batch job."""
    if api_key is None:
        api_key = get_api_key()

    client = OpenAI(api_key=api_key)

    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        msg = f"Batch is not completed yet. Current status: {batch.status}"
        raise ValueError(msg)

    if not batch.output_file_id:
        msg = "No output file available for this batch"
        raise ValueError(msg)

    batch_name = batch.metadata.get("batch_name", batch_id)

    output_content = client.files.content(batch.output_file_id)
    output_path = Path(output_dir) / f"{batch_name}_results.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines_written = 0
    with output_path.open("w") as f:
        for line in output_content.text.strip().split("\n"):
            if line.strip():
                response_obj = json.loads(line)
                filtered_response = {
                    "custom_id": response_obj.get("custom_id"),
                    "response": response_obj.get("response"),
                    "error": response_obj.get("error"),
                }
                f.write(json.dumps(filtered_response) + "\n")
                lines_written += 1

    # Download error file if exists
    if batch.error_file_id:
        error_content = client.files.content(batch.error_file_id)
        error_path = Path(output_dir) / f"{batch_name}_errors.jsonl"

        with error_path.open("w") as f:
            f.write(error_content.text)

    return {
        "output_path": str(output_path),
        "batch_name": batch_name,
        "completed_at": datetime.now(UTC).isoformat(),
        "n_responses": lines_written,
    }


def calculate_actual_cost_from_results(results_jsonl_path, model_name):
    """Calculate actual cost from downloaded JSONL results.

    For fine-tuned models, extracts base model from format:
    ft:BASE_MODEL:org:name:version
    """
    # Extract base model for fine-tuned models
    if model_name.startswith("ft:"):
        base_model = model_name.split(":")[1]
    else:
        base_model = model_name

    if base_model not in OPENAI_API_MODELS:
        return None

    model_config = OPENAI_API_MODELS[base_model]

    # Use fine-tuned pricing if available, otherwise regular pricing
    input_cost_per_1m = (
        model_config.get("fine_tuned_input_cost", model_config["input_cost"])
        * BATCH_API_DISCOUNT
    )
    output_cost_per_1m = (
        model_config.get("fine_tuned_output_cost", model_config["output_cost"])
        * BATCH_API_DISCOUNT
    )

    total_input_tokens = 0
    total_output_tokens = 0

    with Path(results_jsonl_path).open() as f:
        for line in f:
            if not line.strip():
                continue

            result = json.loads(line)

            if result.get("response"):
                usage = result["response"].get("body", {}).get("usage", {})
                total_input_tokens += usage.get("prompt_tokens", 0)
                total_output_tokens += usage.get("completion_tokens", 0)

    if total_input_tokens == 0 and total_output_tokens == 0:
        return None

    input_cost = (total_input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (total_output_tokens / 1_000_000) * output_cost_per_1m

    return {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "input_cost": round(input_cost, 4),
        "output_cost": round(output_cost, 4),
        "total_cost": round(input_cost + output_cost, 4),
    }


def update_metadata_csv(metadata_csv_path, batch_name, updates):
    """Update metadata CSV with new batch information."""
    if not Path(metadata_csv_path).exists():
        return None

    df = pd.read_csv(metadata_csv_path)

    idx = df[df["batch_name"] == batch_name].index
    if len(idx) == 0:
        return None

    idx = idx[0]

    for key, value in updates.items():
        if key in df.columns:
            df.loc[idx, key] = value

    df.to_csv(metadata_csv_path, index=False)

    return df


def submit_batch(jsonl_path, experiment_config, metadata_csv_path, api_key=None):
    """Submit a single ISCO classification batch job (upload + create batch).

    Args:
        jsonl_path: Path to JSONL file (e.g., isco_classification_001.jsonl)
        experiment_config: Experiment configuration dict
        metadata_csv_path: Path to metadata CSV (isco_classification_metadata.csv)
        api_key: OpenAI API key (if None, gets from environment)

    Returns:
        dict: Submission information (batch_id, file_id, batch_name)
    """
    # Extract batch name from filename
    # (e.g., isco_classification_001.jsonl -> isco_classification_001)
    jsonl_file = Path(jsonl_path)
    batch_name = jsonl_file.stem  # Remove .jsonl extension

    # Step 1: Upload file
    file_id = upload_batch_file(jsonl_path, api_key)

    # Step 2: Create batch
    batch_info = create_batch(file_id, experiment_config, batch_name, api_key)

    # Step 3: Update metadata
    if metadata_csv_path:
        updates = {
            "batch_id": batch_info["batch_id"],
            "status": "submitted",
            "submitted_at": batch_info["submitted_at"],
        }
        update_metadata_csv(metadata_csv_path, batch_name, updates)

    return {
        "batch_id": batch_info["batch_id"],
        "file_id": file_id,
        "batch_name": batch_name,
    }


def check_status(batch_id, api_key=None):
    """Check and display batch status."""
    status_info = check_batch_status(batch_id, api_key)

    return status_info


def cancel_batch(batch_id, metadata_csv_path, api_key=None):
    """Cancel a batch job."""
    if api_key is None:
        api_key = get_api_key()

    client = OpenAI(api_key=api_key)

    batch = client.batches.cancel(batch_id)
    batch_name = batch.metadata.get("batch_name", batch_id)

    if metadata_csv_path:
        updates = {"status": "cancelled"}
        update_metadata_csv(metadata_csv_path, batch_name, updates)

    return True
