"""Manager for OpenAI fine-tuning jobs for ISCO classifier."""

import time
from pathlib import Path

import pandas as pd


def upload_file(client, file_path, purpose="fine-tune"):
    """Upload a file to OpenAI.

    Args:
        client: OpenAI client instance
        file_path: Path to the file to upload
        purpose: Purpose of the file

    Returns:
        str: File ID from OpenAI
    """
    with Path(file_path).open("rb") as f:
        response = client.files.create(file=f, purpose=purpose)
    return response.id


def create_fine_tune_job(
    client,
    training_file_id,
    config,
    validation_file_id=None,
):
    """Create a fine-tuning job.

    Args:
        client: OpenAI client instance
        training_file_id: OpenAI file ID for training data
        config: Configuration dictionary
        validation_file_id: OpenAI file ID for validation data (optional)

    Returns:
        dict: Fine-tuning job information
    """
    hyperparameters = {
        "n_epochs": config.get("n_epochs", 3),
    }

    # Only add batch_size and learning_rate_multiplier if not "auto"
    if config.get("batch_size") != "auto":
        hyperparameters["batch_size"] = config["batch_size"]
    if config.get("learning_rate_multiplier") != "auto":
        hyperparameters["learning_rate_multiplier"] = config["learning_rate_multiplier"]

    kwargs = {
        "training_file": training_file_id,
        "model": config["model"],
        "hyperparameters": hyperparameters,
    }

    if validation_file_id:
        kwargs["validation_file"] = validation_file_id

    if config.get("model_suffix"):
        kwargs["suffix"] = config["model_suffix"]

    response = client.fine_tuning.jobs.create(**kwargs)

    return {
        "job_id": response.id,
        "status": response.status,
        "model": response.model,
        "created_at": response.created_at,
        "training_file": response.training_file,
        "validation_file": getattr(response, "validation_file", None),
    }


def get_job_status(client, job_id):
    """Get the status of a fine-tuning job.

    Args:
        client: OpenAI client instance
        job_id: OpenAI fine-tuning job ID

    Returns:
        dict: Job status information
    """
    response = client.fine_tuning.jobs.retrieve(job_id)

    return {
        "job_id": response.id,
        "status": response.status,
        "fine_tuned_model": response.fine_tuned_model,
        "trained_tokens": response.trained_tokens,
        "error": response.error if hasattr(response, "error") else None,
    }


def wait_for_completion(client, job_id, check_interval=60):
    """Wait for a fine-tuning job to complete.

    Args:
        client: OpenAI client instance
        job_id: OpenAI fine-tuning job ID
        check_interval: Seconds between status checks

    Returns:
        dict: Final job status
    """
    while True:
        status_info = get_job_status(client, job_id)
        status = status_info["status"]

        if status in ["succeeded", "failed", "cancelled"]:
            return status_info

        time.sleep(check_interval)


def list_fine_tune_jobs(client, limit=10):
    """List recent fine-tuning jobs.

    Args:
        client: OpenAI client instance
        limit: Maximum number of jobs to return

    Returns:
        list: List of job information dictionaries
    """
    response = client.fine_tuning.jobs.list(limit=limit)

    jobs = [
        {
            "job_id": job.id,
            "status": job.status,
            "model": job.model,
            "fine_tuned_model": job.fine_tuned_model,
            "created_at": job.created_at,
        }
        for job in response.data
    ]

    return jobs


def save_job_info(job_info, metadata_path):
    """Save or update fine-tuning job information to metadata CSV.

    Args:
        job_info: Dictionary with job information
        metadata_path: Path to the metadata CSV file
    """
    metadata_path = Path(metadata_path)

    # Load existing metadata
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
    else:
        return

    # Add job info columns
    metadata["job_id"] = job_info["job_id"]
    metadata["job_status"] = job_info["status"]
    metadata["fine_tuned_model"] = job_info.get("fine_tuned_model")
    metadata["trained_tokens"] = job_info.get("trained_tokens")

    # Save updated metadata
    metadata.to_csv(metadata_path, index=False)
