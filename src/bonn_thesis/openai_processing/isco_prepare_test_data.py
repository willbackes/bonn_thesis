"""Functions for preparing ISCO classification test data."""

import random


def sample_test_requests(all_requests, n_samples=300, random_seed=42):
    """Randomly sample requests for test data.

    Args:
        all_requests: List of all classification requests
        n_samples: Number of samples to select
        random_seed: Random seed for reproducibility

    Returns:
        list: Sampled requests
    """
    random.seed(random_seed)

    # Ensure we don't sample more than available
    n_samples = min(n_samples, len(all_requests))

    return random.sample(all_requests, n_samples)
