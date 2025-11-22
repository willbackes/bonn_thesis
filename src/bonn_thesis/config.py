"""All the general configuration of the project."""

from pathlib import Path

SRC = Path(__file__).parent.resolve()
ROOT = SRC.joinpath("..", "..").resolve()

BLD = ROOT.joinpath("bld").resolve()
DOCUMENTS = ROOT.joinpath("documents").resolve()

RAW_DATA = SRC.joinpath("data", "raw").resolve()
RAW_DATA_BLD = BLD.joinpath("data", "raw").resolve()

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

BATCH_SIZE = 10000

CITIES_ENGLISH = [
    {"bland_code": "05", "reg_code": "053150000000", "plz": "50667", "city": "Cologne"},
    {"bland_code": "09", "reg_code": "091620000000", "plz": "80331", "city": "Munich"},
    {
        "bland_code": "06",
        "reg_code": "064120000000",
        "plz": "60311",
        "city": "Frankfurt",
    },
    {
        "bland_code": "05",
        "reg_code": "051110000000",
        "plz": "40213",
        "city": "Dusseldorf",
    },
    {"bland_code": "03", "reg_code": "032410001001", "plz": "30159", "city": "Hanover"},
    {
        "bland_code": "08",
        "reg_code": "083110000000",
        "plz": "79098",
        "city": "Freiburg",
    },
]

BUNDESLAND_MAP = {
    "01": ["Schleswig-Holstein", "Schleswig-Holstein"],
    "02": ["Hamburg", "Hamburg"],
    "03": ["Niedersachsen", "Lower Saxony"],
    "04": ["Bremen", "Bremen"],
    "05": ["Nordrhein-Westfalen", "North Rhine-Westphalia"],
    "06": ["Hessen", "Hesse"],
    "07": ["Rheinland-Pfalz", "Rhineland-Palatinate"],
    "08": ["Baden-Württemberg", "Baden-Württemberg"],
    "09": ["Bayern", "Bavaria"],
    "10": ["Saarland", "Saarland"],
    "11": ["Berlin", "Berlin"],
    "12": ["Brandenburg", "Brandenburg"],
    "13": ["Mecklenburg-Vorpommern", "Mecklenburg-West Pomerania"],
    "14": ["Sachsen", "Saxony"],
    "15": ["Sachsen-Anhalt", "Saxony-Anhalt"],
    "16": ["Thüringen", "Thuringia"],
}
