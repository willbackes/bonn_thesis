from collections import Counter

import pandas as pd


def clean_degree_type(series: pd.Series) -> pd.Series:
    """Pre-clean degree_type strings."""
    cleaned = (
        series.astype(str)
        .str.lower()
        .str.replace("ä", "a", regex=False)
        .str.replace("ö", "o", regex=False)
        .str.replace("ü", "u", regex=False)
        .str.replace("ß", "ss", regex=False)
        .str.replace("é", "e", regex=False)
        .str.replace("è", "e", regex=False)
        .str.replace("ê", "e", regex=False)
        .str.replace("à", "a", regex=False)
        .str.replace("â", "a", regex=False)
        .str.replace("ç", "c", regex=False)
        .str.replace("ñ", "n", regex=False)
        .str.replace("ã", "a", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"\.", "", regex=True)
        .str.replace(r"[^\w\s-]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return cleaned


def aggregate_unclassified_degrees(education_data: pd.DataFrame) -> Counter:
    """Aggregate degree_type frequencies for unclassified records.

    Args:
        education_data: DataFrame with education data.

    Returns:
        Counter with degree_type frequencies
    """
    degree_type_counter = Counter()

    unclassified = education_data[
        (education_data["case_degree_label"].isna())
        & (education_data["start_date"].notna())
        & (education_data["end_date"].notna())
    ].copy()

    if len(unclassified) > 0:
        unclassified["degree_type_clean"] = clean_degree_type(
            unclassified["degree_type"]
        )

        unclassified_filtered = unclassified[
            ~unclassified["degree_type_clean"].isin(["nan", "none", "null", "na", ""])
        ]

        degree_counts = unclassified_filtered["degree_type_clean"].value_counts()
        degree_type_counter.update(degree_counts.to_dict())

    return degree_type_counter


def get_top_n_degrees(counter: Counter, n: int) -> pd.DataFrame:
    """Get top N degree types as DataFrame."""
    degree_type_freq = pd.Series(counter).sort_values(ascending=False)
    top_n = degree_type_freq.head(n)

    top_n_df = pd.DataFrame(
        {"degree_type_cleaned": top_n.index, "count": top_n.to_numpy()}
    )

    return top_n_df
