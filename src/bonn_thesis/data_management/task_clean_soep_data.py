"""Tasks for cleaning SOEP survey data."""

import pandas as pd

from bonn_thesis.config import BLD, RAW_DATA, SOEP_DATA
from bonn_thesis.data_management.clean_soep_data import clean_soep_data


def task_clean_soep_data(
    pgen_data=SOEP_DATA / "pgen.dta",
    ppathl_data=SOEP_DATA / "ppathl.dta",
    hbrutto_data=SOEP_DATA / "hbrutto.dta",
    isco_data=RAW_DATA / "ISCO-08 EN Structure and definitions.xlsx",
    produces=BLD / "data" / "soep_clean.parquet",
):
    """Clean and merge SOEP survey data with occupation classifications."""
    pgen = pd.read_stata(
        pgen_data,
        columns=[
            "hid",
            "pid",
            "syear",
            "pgkldb2010",
            "pgisco08",
            "pgisced11",
            "pgexpft",
            "pglabgro",
            "pglabnet",
        ],
    )
    ppathl = pd.read_stata(ppathl_data, columns=["hid", "pid", "syear", "sex"])
    hbrutto = pd.read_stata(hbrutto_data, columns=["hid", "syear", "bula_h"])
    isco = pd.read_excel(isco_data)

    soep_clean = clean_soep_data(pgen, ppathl, hbrutto, isco)
    soep_clean.to_parquet(produces)
