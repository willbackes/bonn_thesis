"""All the general configuration of the project."""

from pathlib import Path

SRC = Path(__file__).parent.resolve()
ROOT = SRC.joinpath("..", "..").resolve()

BLD = ROOT.joinpath("bld").resolve()
DOCUMENTS = ROOT.joinpath("documents").resolve()

RAW_DATA = BLD.joinpath("data", "raw").resolve()

TEMPLATE_GROUPS = ["marital_status", "highest_qualification"]

config_database = {
    "database": {
        "host": "localhost",
        "user": "postgres",
        "password": "2217",
        "database": "experience-scoring",
        "port": "5432",
    }
}

BATCH_SIZE = 50000
