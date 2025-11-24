"""Utilities for matching and normalizing German location data."""

import re

import pandas as pd

from bonn_thesis.config import CITY_BLACKLIST, FIRST_WORD_MIN_LENGTH
from bonn_thesis.data_management.location_normalize_string import (
    normalize_location_string,
)


def clean_location_data(
    location: pd.DataFrame, bundesland_reference: pd.DataFrame
) -> pd.DataFrame:
    """Add federal state information to experience DataFrame.

    Extracts German federal state and city from 'exp_location' column using
    a hierarchical matching strategy. Creates three new columns in the DataFrame.

    Args:
        location: DataFrame with 'exp_location' column
        bundesland_reference: DataFrame with reference location data
    Returns:
        DataFrame with additional columns:
        - 'matched_state': German federal state name (or None)
        - 'matched_city': Matched city name (or None)
        - 'match_method': How the match was made (matching strategy used)
    """
    clean_location = pd.DataFrame()

    clean_location["exp_id"] = location["exp_id"].astype(pd.Int32Dtype())
    clean_location["prof_id"] = location["prof_id"].astype(pd.Int32Dtype())
    clean_location["comp_id"] = location["comp_id"].astype(pd.Int32Dtype())
    clean_location["industry_id"] = location["industry_id"].astype(pd.Int32Dtype())
    clean_location["job_title_id"] = location["job_title_id"].astype(pd.Int32Dtype())
    clean_location["exp_location"] = location["exp_location"]

    city_lookup = _build_city_lookup(bundesland_reference)
    state_en_lookup, state_de_lookup, state_en_list, state_de_list = (
        _build_state_lookups(bundesland_reference)
    )

    results = location["exp_location"].apply(
        lambda x: identify_german_federal_state(
            x,
            city_lookup,
            state_en_lookup,
            state_de_lookup,
            state_en_list,
            state_de_list,
        )
    )

    clean_location["matched_city"] = results.apply(lambda x: x["city"])
    clean_location["matched_state"] = results.apply(lambda x: x["state"])
    state_bland_map = (
        bundesland_reference[["state_en", "bland_code"]]
        .drop_duplicates()
        .set_index("state_en")["bland_code"]
        .to_dict()
    )
    clean_location["bland_code"] = (
        clean_location["matched_state"].map(state_bland_map).astype("Int32")
    )
    clean_location["match_method"] = results.apply(lambda x: x["method"])

    return clean_location


def identify_german_federal_state(
    location: str,
    city_lookup: dict,
    state_en_lookup: pd.DataFrame,
    state_de_lookup: pd.DataFrame,
    state_en_list: list,
    state_de_list: list,
) -> dict:
    """Extract German federal state from location string.

    Uses ordered matching strategy:
    1. Exact state match (EN, then DE)
    2. Exact city match (full string or first word)
    3. Substring state match (EN, then DE)
    4. Substring city match (word boundary, longest first)

    Args:
        location: Raw location string from LinkedIn
        city_lookup: Dict mapping normalized cities to (city, state) tuples
        state_en_lookup: DataFrame with English state names
        state_de_lookup: DataFrame with German state names
        state_en_list: Sorted list of normalized English state names
        state_de_list: Sorted list of normalized German state names

    Returns:
        Dictionary with keys: 'state', 'city', 'method'
    """
    # Handle missing values
    if pd.isna(location) or location == "":
        return {"state": None, "city": None, "method": "missing"}

    # Normalize location string
    normalized = normalize_location_string(location)

    # Try matching strategies in order
    matchers = [
        (_exact_match_state, (normalized, state_en_lookup, "en")),
        (_exact_match_state, (normalized, state_de_lookup, "de")),
        (_exact_match_city, (normalized, city_lookup)),
        (_substring_match_state, (normalized, state_en_list, state_en_lookup, "en")),
        (_substring_match_state, (normalized, state_de_list, state_de_lookup, "de")),
        (_substring_match_city, (normalized, city_lookup)),
    ]

    for matcher, args in matchers:
        result = matcher(*args)
        if result:
            return result

    # No match found
    return {"state": None, "city": None, "method": "no_match"}


def _identify_germany_string(normalized_location: str) -> bool:
    """Check if location mentions Germany-related terms."""
    germany_keywords = ["germany", "deutschland", "allemagne", "de", "ger"]
    return any(keyword in normalized_location for keyword in germany_keywords)


def _build_city_lookup(reference_df: pd.DataFrame) -> dict:
    """Build city lookup dictionary for fast access during matching."""
    return dict(
        zip(
            reference_df["city_normalized"],
            zip(reference_df["city"], reference_df["state_en"], strict=False),
            strict=False,
        )
    )


def _build_state_lookups(reference_df: pd.DataFrame) -> tuple:
    """Build state lookup dictionaries and sorted lists for efficient matching."""
    state_en_lookup = reference_df[
        ["state_en", "state_en_normalized"]
    ].drop_duplicates()
    state_de_lookup = reference_df[
        ["state_en", "state_de_normalized"]
    ].drop_duplicates()

    state_en_list = sorted(
        state_en_lookup["state_en_normalized"].tolist(), key=len, reverse=True
    )
    state_de_list = sorted(
        state_de_lookup["state_de_normalized"].tolist(), key=len, reverse=True
    )

    return state_en_lookup, state_de_lookup, state_en_list, state_de_list


def _exact_match_state(
    normalized_location: str, state_lookup: pd.DataFrame, state_type: str
) -> dict:
    """Perform exact state name matching."""
    state_col = "state_en_normalized" if state_type == "en" else "state_de_normalized"

    for _, row in state_lookup.iterrows():
        if row[state_col] == normalized_location:
            return {
                "state": row["state_en"],
                "city": None,
                "method": f"exact_state_match_{state_type}",
            }

    return None


def _exact_match_city(normalized_location: str, city_lookup: dict) -> dict:
    """Perform exact city matching including first-word fallback."""
    # Strategy 1: Full string exact match
    if normalized_location in city_lookup:
        city, state = city_lookup[normalized_location]
        return {"state": state, "city": city, "method": "exact_city_match"}

    # Strategy 2: First word exact match (for blacklisted cities like "Rain", "Ulm")
    words = normalized_location.split()
    if words:
        first_word = words[0]
        if first_word in city_lookup and len(first_word) >= FIRST_WORD_MIN_LENGTH:
            city, state = city_lookup[first_word]
            return {
                "state": state,
                "city": city,
                "method": "exact_city_match_first_word",
            }

    return None


def _substring_match_state(
    normalized_location: str,
    state_list: list,
    state_lookup: pd.DataFrame,
    state_type: str,
) -> dict:
    """Perform substring state matching with word boundaries."""
    state_col = "state_en_normalized" if state_type == "en" else "state_de_normalized"

    for state_norm in state_list:
        if state_norm in normalized_location and state_norm != "":
            pattern = r"\b" + re.escape(state_norm) + r"\b"
            if re.search(pattern, normalized_location):
                state_en = state_lookup[state_lookup[state_col] == state_norm][
                    "state_en"
                ].iloc[0]
                return {
                    "state": state_en,
                    "city": None,
                    "method": f"substring_state_match_{state_type}",
                }

    return None


def _substring_match_city(normalized_location: str, city_lookup: dict) -> dict:
    """Perform substring city matching prioritizing longer city names."""
    city_list_sorted = sorted(
        [c for c in city_lookup if c != "" and c not in CITY_BLACKLIST],
        key=len,
        reverse=True,
    )

    for city_norm in city_list_sorted:
        if city_norm in normalized_location and city_norm != "":
            pattern = r"\b" + re.escape(city_norm) + r"\b"
            if re.search(pattern, normalized_location):
                city, state = city_lookup[city_norm]
                return {"state": state, "city": city, "method": "substring_city_match"}

    return None
