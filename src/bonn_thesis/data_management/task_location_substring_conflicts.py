"""Task for identifying location substring conflicts."""

import pandas as pd

from bonn_thesis.config import BLD
from bonn_thesis.data_management.location_substring_conflicts import (
    identify_substring_conflicts,
)


def task_identify_location_substring_conflicts(
    data=BLD / "data" / "bundesland_reference.parquet",
    produces=BLD / "data" / "substring_conflicts.parquet",
):
    """Identify cities that are substrings of other cities.

    This helps find matching conflicts like:
    - "Homburg" is a substring of "Bad Homburg v. d. Höhe"
    - "Burg" is a substring of "Marburg"

    Args:
        data: Input bundesland reference data
        produces: Output conflicts data
    """
    reference_df = pd.read_parquet(data)

    conflicts_df = identify_substring_conflicts(reference_df)

    conflicts_df.to_parquet(produces, index=False)
