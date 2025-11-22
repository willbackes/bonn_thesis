"""Identify location substring conflicts using bundesland reference data."""

import re

import pandas as pd


def identify_substring_conflicts(reference_df: pd.DataFrame) -> pd.DataFrame:
    """Identify cities that are substrings of other cities.

    This helps find cases like:
    - "Homburg" is a substring of "Bad Homburg v. d. Höhe"
    - "Burg" is a substring of "Marburg"

    Returns:
        DataFrame with conflicts
    """
    cities_normalized = reference_df["city_normalized"].dropna().unique()

    conflicts = []

    for city1 in cities_normalized:
        for city2 in cities_normalized:
            if city1 != city2 and len(city1) < len(city2):
                # Check if city1 is a word within city2
                pattern = r"\b" + re.escape(city1) + r"\b"
                if re.search(pattern, city2):
                    # Get the original city names
                    orig_city1 = reference_df[reference_df["city_normalized"] == city1][
                        "city"
                    ].iloc[0]
                    orig_city2 = reference_df[reference_df["city_normalized"] == city2][
                        "city"
                    ].iloc[0]

                    conflicts.append(
                        {
                            "shorter_city": orig_city1,
                            "shorter_normalized": city1,
                            "longer_city": orig_city2,
                            "longer_normalized": city2,
                        }
                    )

    return pd.DataFrame(conflicts)
