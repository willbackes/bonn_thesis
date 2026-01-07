"""Functions for preparing JSONL files for OpenAI fine-tuning of ISCO classifier."""

import json


def split_train_validation(data, split_ratio=0.8, random_seed=42):
    """Split data into training and validation sets.

    Args:
        data: DataFrame with input/output data
        split_ratio: Ratio of data to use for training
        random_seed: Random seed for reproducibility

    Returns:
        tuple: (train_data, validation_data)
    """
    data_shuffled = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    split_idx = int(len(data_shuffled) * split_ratio)

    train_data = data_shuffled[:split_idx]
    val_data = data_shuffled[split_idx:]

    return train_data, val_data


def create_fine_tune_jsonl(data, input_col, output_col, system_message, output_path):
    """Create a JSONL file for fine-tuning in OpenAI format.

    Args:
        data: DataFrame with input/output columns
        input_col: Column name for input (job title)
        output_col: Column name for output (ISCO code)
        system_message: System message for the assistant
        output_path: Path to save the JSONL file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        for _, row in data.iterrows():
            example = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": str(row[input_col])},
                    {"role": "assistant", "content": str(row[output_col])},
                ]
            }
            f.write(json.dumps(example) + "\n")
