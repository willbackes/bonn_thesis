"""Tasks for cleaning German city/state reference data."""

import pandas as pd

from bonn_thesis.config import BLD, SRC
from bonn_thesis.data_management.clean_bundesland_data import clean_bundesland_data


def task_clean_bundesland_data(
    script=SRC / "data_management" / "clean_bundesland_data.py",
    data=SRC / "data" / "raw" / "05-staedte.xlsx",
    produces=BLD / "data" / "bundesland_reference.parquet",
):
    """Clean the German city and state reference data."""
    data = pd.read_excel(data, sheet_name="Städte", skiprows=1, skipfooter=9)
    data = clean_bundesland_data(data)
    data.to_parquet(produces)
