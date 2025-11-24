"""Functions to normalize location strings for matching."""

import re

import pandas as pd


def normalize_location_string(location: str) -> str:
    """Normalize location string for matching.

    Args:
        location: Raw location string from LinkedIn or reference Bundesland data
    Returns:
        Normalized location string.
    """
    if pd.isna(location):
        return ""

    location = str(location).lower().strip()

    umlaut_map = str.maketrans(
        {
            "ä": "ae",
            "ö": "oe",
            "ü": "ue",  # codespell:ignore
            "ß": "ss",
            "Ä": "ae",
            "Ö": "oe",
            "Ü": "ue",  # codespell:ignore
        }
    )
    location = location.translate(umlaut_map)

    location = re.sub(r"\bund umgebung\b", "", location)
    location = re.sub(r"\barea\b", "", location)
    location = re.sub(r"\bregion\b", "", location)
    location = re.sub(r"\bmetropolitan\b", "", location)
    location = re.sub(r"\bgreater\b", "", location)

    location = re.sub(r"[^\w\s\-]", " ", location)
    location = re.sub(r"/", " ", location)

    location = re.sub(r"\s+", " ", location)

    return location.strip()
