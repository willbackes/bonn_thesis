"""Functions for estimating costs of OpenAI fine-tuning."""

import json

import tiktoken

from bonn_thesis.config import DEFAULT_OPENAI_MODEL, OPENAI_API_MODELS


def count_fine_tune_tokens(jsonl_path, encoding_name="cl100k_base"):
    """Count tokens in fine-tuning dataset.

    Args:
        jsonl_path: Path to the JSONL training or validation file
        encoding_name: Tiktoken encoding name

    Returns:
        dict: Token counts and statistics
    """
    encoding = tiktoken.get_encoding(encoding_name)

    total_tokens = 0
    n_examples = 0

    with jsonl_path.open() as f:
        for line in f:
            example = json.loads(line)
            messages = example.get("messages", [])

            # Count tokens for all messages (system, user, assistant)
            for message in messages:
                content = message.get("content", "")
                content_tokens = len(encoding.encode(content))
                # Add 4 tokens overhead per message for message formatting
                total_tokens += content_tokens + 4

            # Add 3 tokens per example for formatting overhead
            total_tokens += 3
            n_examples += 1

    return {
        "n_examples": n_examples,
        "total_tokens": total_tokens,
        "avg_tokens_per_example": total_tokens / n_examples if n_examples > 0 else 0,
    }


def calculate_fine_tune_training_cost(
    training_token_counts,
    validation_token_counts,
    model_pricing,
    n_epochs=3,
):
    """Calculate fine-tuning training costs.

    Args:
        training_token_counts: Token counts for training data
        validation_token_counts: Token counts for validation data
        model_pricing: Pricing dictionary with training_cost
        n_epochs: Number of training epochs

    Returns:
        dict: Training cost estimates
    """
    # Training cost is per 1M tokens
    training_tokens_total = training_token_counts["total_tokens"] * n_epochs
    training_cost = (training_tokens_total / 1_000_000) * model_pricing["training"]

    # Validation is run once per epoch but not charged (included in training)
    validation_tokens = (
        validation_token_counts.get("total_tokens", 0) if validation_token_counts else 0
    )

    return {
        "training_tokens": training_token_counts["total_tokens"],
        "training_tokens_total": training_tokens_total,
        "validation_tokens": validation_tokens,
        "training_cost": round(training_cost, 4),
        "n_epochs": n_epochs,
    }


def calculate_fine_tune_inference_cost(
    n_predictions,
    avg_input_tokens,
    avg_output_tokens,
    model_pricing,
):
    """Calculate estimated inference costs for fine-tuned model.

    Args:
        n_predictions: Number of predictions to estimate for
        avg_input_tokens: Average input tokens per prediction
        avg_output_tokens: Average output tokens per prediction
        model_pricing: Pricing dictionary


    Returns:
        dict: Inference cost estimates
    """
    total_input_tokens = n_predictions * avg_input_tokens
    total_output_tokens = n_predictions * avg_output_tokens

    input_cost = (total_input_tokens / 1_000_000) * model_pricing["fine_tuned_input"]
    output_cost = (total_output_tokens / 1_000_000) * model_pricing["fine_tuned_output"]

    return {
        "n_predictions": n_predictions,
        "estimated_input_tokens": total_input_tokens,
        "estimated_output_tokens": total_output_tokens,
        "estimated_input_cost": round(input_cost, 4),
        "estimated_output_cost": round(output_cost, 4),
        "estimated_total_inference_cost": round(input_cost + output_cost, 4),
        "cost_per_prediction": round((input_cost + output_cost) / n_predictions, 6)
        if n_predictions > 0
        else 0,
    }


def estimate_fine_tune_costs(
    training_jsonl_path,
    validation_jsonl_path,
    config,
    n_predictions_to_estimate=1000,
):
    """Estimate all costs for fine-tuning project.

    Args:
        training_jsonl_path: Path to training JSONL file
        validation_jsonl_path: Path to validation JSONL file (can be None)
        config: Configuration dictionary with model and n_epochs
        n_predictions_to_estimate: Number of predictions to estimate inference cost for

    Returns:
        dict: Complete cost estimates including training and inference

    Raises:
        ValueError: If model is not configured in OPENAI_API_MODELS
    """
    model_name = config.get("model", DEFAULT_OPENAI_MODEL)

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

    # Check if model supports fine-tuning
    if "training_cost" not in model_config:
        msg = f"Model '{model_name}' does not have fine-tuning pricing configured"
        raise ValueError(msg)

    model_pricing = {
        "training": model_config["training_cost"],
        "fine_tuned_input": model_config["fine_tuned_input_cost"],
        "fine_tuned_output": model_config["fine_tuned_output_cost"],
    }

    # Count tokens
    training_counts = count_fine_tune_tokens(training_jsonl_path)
    validation_counts = (
        count_fine_tune_tokens(validation_jsonl_path)
        if validation_jsonl_path and validation_jsonl_path.exists()
        else None
    )

    # Calculate training costs
    training_costs = calculate_fine_tune_training_cost(
        training_token_counts=training_counts,
        validation_token_counts=validation_counts,
        model_pricing=model_pricing,
        n_epochs=config.get("n_epochs", 3),
    )

    # Estimate inference costs
    # Use average input tokens from training data as proxy
    # Assume output is ISCO code (roughly 3-5 tokens)
    avg_output_tokens = 5  # ISCO 3-digit code + formatting
    inference_costs = calculate_fine_tune_inference_cost(
        n_predictions=n_predictions_to_estimate,
        avg_input_tokens=training_counts["avg_tokens_per_example"]
        / 3,  # Divide by 3 for just user message
        avg_output_tokens=avg_output_tokens,
        model_pricing=model_pricing,
    )

    # Combine all estimates
    return {
        "fine_tune_id": config.get("fine_tune_id", ""),
        "model": model_name,
        "n_training_examples": training_counts["n_examples"],
        "n_validation_examples": validation_counts["n_examples"]
        if validation_counts
        else 0,
        "training_tokens_per_example": round(
            training_counts["avg_tokens_per_example"], 2
        ),
        "training_tokens_total": training_costs["training_tokens_total"],
        "training_cost": training_costs["training_cost"],
        "n_epochs": training_costs["n_epochs"],
        "inference_cost_per_1k": round(
            inference_costs["cost_per_prediction"] * 1000, 4
        ),
        "inference_cost_per_10k": round(
            inference_costs["cost_per_prediction"] * 10000, 4
        ),
        "inference_cost_per_100k": round(
            inference_costs["cost_per_prediction"] * 100000, 4
        ),
    }


def count_classification_tokens(jsonl_path, encoding_name="cl100k_base"):
    """Count tokens in classification JSONL file.

    Args:
        jsonl_path: Path to the JSONL file with classification requests
        encoding_name: Tiktoken encoding name

    Returns:
        dict: Token counts and statistics
    """
    encoding = tiktoken.get_encoding(encoding_name)

    total_input_tokens = 0
    n_requests = 0

    with jsonl_path.open() as f:
        for line in f:
            if not line.strip():  # Skip empty lines
                continue

            request = json.loads(line)
            messages = request.get("body", {}).get("messages", [])

            # Count tokens for system and user messages (input)
            for message in messages:
                if message.get("role") in ["system", "user"]:
                    content = message.get("content", "")
                    content_tokens = len(encoding.encode(content))
                    # Add 4 tokens overhead per message for formatting
                    total_input_tokens += content_tokens + 4

            # Add 3 tokens per request for formatting overhead
            total_input_tokens += 3
            n_requests += 1

    return {
        "n_requests": n_requests,
        "total_input_tokens": total_input_tokens,
        "avg_input_tokens": total_input_tokens / n_requests if n_requests > 0 else 0,
    }


def estimate_classification_batch_costs(
    jsonl_path,
    config,
    avg_output_tokens=3,
    batch_api_discount=0.5,
):
    """Estimate costs for a single classification batch.

    Args:
        jsonl_path: Path to JSONL file with classification requests
        config: Configuration dictionary with model info
        avg_output_tokens: Expected average output tokens (default 3 for ISCO codes)
        batch_api_discount: Batch API discount factor (default 0.5 = 50% off)

    Returns:
        dict: Cost estimates for the batch
    """
    from pathlib import Path

    jsonl_path = Path(jsonl_path)

    # Get model name from config
    model_name = config.get("model", DEFAULT_OPENAI_MODEL)

    # Extract base model name if it's a fine-tuned model
    # Format: ft:BASE_MODEL:org:name:version
    if model_name.startswith("ft:"):
        base_model = model_name.split(":")[1]
    else:
        base_model = model_name

    # Get pricing
    if base_model not in OPENAI_API_MODELS:
        available_models = ", ".join(OPENAI_API_MODELS.keys())
        msg = f"Base model '{base_model}' not configured. Available: {available_models}"
        raise ValueError(msg)

    model_config = OPENAI_API_MODELS[base_model]
    model_pricing = {
        "fine_tuned_input": model_config.get(
            "fine_tuned_input_cost", model_config.get("input_cost")
        ),
        "fine_tuned_output": model_config.get(
            "fine_tuned_output_cost", model_config.get("output_cost")
        ),
    }

    # Count tokens
    token_counts = count_classification_tokens(jsonl_path)

    # Calculate costs
    inference_costs = calculate_fine_tune_inference_cost(
        n_predictions=token_counts["n_requests"],
        avg_input_tokens=token_counts["avg_input_tokens"],
        avg_output_tokens=avg_output_tokens,
        model_pricing=model_pricing,
    )

    # Apply batch API discount
    total_cost_before_discount = inference_costs["estimated_total_inference_cost"]
    total_cost_after_discount = total_cost_before_discount * batch_api_discount

    # Extract batch info from filename
    batch_name = jsonl_path.stem  # e.g., isco_classification_001

    return {
        "batch_name": batch_name,
        "jsonl_file": jsonl_path.name,
        "n_requests": token_counts["n_requests"],
        "avg_input_tokens": round(token_counts["avg_input_tokens"], 2),
        "avg_output_tokens": avg_output_tokens,
        "total_input_tokens": token_counts["total_input_tokens"],
        "total_output_tokens": token_counts["n_requests"] * avg_output_tokens,
        "estimated_cost_before_discount": round(total_cost_before_discount, 4),
        "batch_api_discount": batch_api_discount,
        "estimated_cost_after_discount": round(total_cost_after_discount, 4),
        "cost_per_request": round(
            total_cost_after_discount / token_counts["n_requests"], 6
        )
        if token_counts["n_requests"] > 0
        else 0,
    }


def estimate_all_classification_costs(
    input_dir,
    config,
    avg_output_tokens=3,
    batch_api_discount=0.5,
):
    """Estimate costs for all classification batches in a directory.

    Args:
        input_dir: Directory containing JSONL files
        config: Configuration dictionary with model info
        avg_output_tokens: Expected average output tokens
        batch_api_discount: Batch API discount factor

    Returns:
        pd.DataFrame: Cost estimates for all batches
    """
    from pathlib import Path

    import pandas as pd

    input_dir = Path(input_dir)
    jsonl_files = sorted(input_dir.glob("isco_classification_*.jsonl"))

    results = []
    for jsonl_file in jsonl_files:
        # Skip empty files
        if jsonl_file.stat().st_size == 0:
            continue

        batch_costs = estimate_classification_batch_costs(
            jsonl_path=jsonl_file,
            config=config,
            avg_output_tokens=avg_output_tokens,
            batch_api_discount=batch_api_discount,
        )
        results.append(batch_costs)

    df = pd.DataFrame(results)

    # Add summary row
    if len(df) > 0:
        summary = {
            "batch_name": "TOTAL",
            "jsonl_file": f"{len(df)} files",
            "n_requests": df["n_requests"].sum(),
            "avg_input_tokens": df["avg_input_tokens"].mean(),
            "avg_output_tokens": avg_output_tokens,
            "total_input_tokens": df["total_input_tokens"].sum(),
            "total_output_tokens": df["total_output_tokens"].sum(),
            "estimated_cost_before_discount": df[
                "estimated_cost_before_discount"
            ].sum(),
            "batch_api_discount": batch_api_discount,
            "estimated_cost_after_discount": df["estimated_cost_after_discount"].sum(),
            "cost_per_request": df["estimated_cost_after_discount"].sum()
            / df["n_requests"].sum()
            if df["n_requests"].sum() > 0
            else 0,
        }
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    return df
