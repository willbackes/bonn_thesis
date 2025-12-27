"""Functions for preparing JSONL files for OpenAI batch API."""

import json


def format_prompt(row, prompt_config):
    """Format a prompt for a single row using the prompt template.

    Args:
        row: DataFrame row (as Series)
        prompt_config: Prompt configuration dictionary

    Returns:
        str: Formatted prompt
    """
    template = prompt_config["prompt_template"]
    required_vars = prompt_config.get("required_variables", [])

    # Build format dictionary from row
    format_dict = {}
    for var in required_vars:
        if var in row.index:
            value = row[var]
            # Format numeric values to 2 decimal places
            if isinstance(value, int | float) and var == "pgexpft_mean":
                format_dict[var] = f"{value:.2f}"
            else:
                format_dict[var] = value
        else:
            msg = f"Required variable '{var}' not found in data"
            raise ValueError(msg)

    return template.format(**format_dict)


def create_jsonl_request(row, prompt_config, experiment_config, row_idx):
    """Create a single JSONL request for OpenAI batch API.

    Args:
        row: DataFrame row (as Series)
        prompt_config: Prompt configuration dictionary
        experiment_config: Experiment configuration dictionary
        row_idx: Row index for unique ID generation

    Returns:
        dict: Request object for JSONL
    """
    batch_name = experiment_config["batch_name"]
    custom_id = f"{batch_name}_{row_idx}"

    # Format the user prompt
    user_prompt = format_prompt(row, prompt_config)

    # Build request
    request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": experiment_config["model"],
            "messages": [
                {"role": "system", "content": prompt_config.get("system_message", "")},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": experiment_config.get("temperature", 0.7),
            "max_tokens": experiment_config.get("max_tokens", 100),
        },
    }

    return request


def save_jsonl(requests, output_path):
    """Save requests to JSONL file.

    Args:
        requests: List of request dictionaries
        output_path: Path to output JSONL file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")


def prepare_jsonl_for_openai(aggregated_df, experiment_config, prompt_config):
    """Main function to prepare JSONL file from aggregated data.

    Args:
        aggregated_df: DataFrame with aggregated data
        experiment_config: Experiment configuration dictionary
        prompt_config: Prompt configuration dictionary

    Returns:
        tuple: (list of request dicts, DataFrame with reference data)
    """
    # Create JSONL requests
    requests = []
    for idx, row in aggregated_df.iterrows():
        request = create_jsonl_request(row, prompt_config, experiment_config, idx)
        requests.append(request)

    return requests, aggregated_df
