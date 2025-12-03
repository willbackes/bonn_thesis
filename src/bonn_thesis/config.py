"""All the general configuration of the project."""

from pathlib import Path

SRC = Path(__file__).parent.resolve()
ROOT = SRC.joinpath("..", "..").resolve()

BLD = ROOT.joinpath("bld").resolve()
DOCUMENTS = ROOT.joinpath("documents").resolve()

RAW_DATA = SRC.joinpath("data", "raw").resolve()
RAW_DATA_BLD = BLD.joinpath("data", "raw").resolve()

SOEP_DATA = ROOT.joinpath("..", "..").resolve().joinpath("datasets", "SOEP_data")

MIN_DATA_YEAR = 2010
MAX_DATA_YEAR = 2019

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

# Reference data for location cleaning

FIRST_WORD_MIN_LENGTH = 3

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
    {
        "bland_code": "06",
        "reg_code": "064340001001",
        "plz": "61343",
        "city": "Bad Homburg",
    },
    {
        "bland_code": "09",
        "reg_code": "091840119119",
        "plz": "85748",
        "city": "Garching bei München",
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

CITY_BLACKLIST = {
    "burg",  # 4 chars - matches Johannesburg, Edinburgh
    "rain",  # 4 chars - matches Ukraine (but "Rain" is a valid German city!)
    "hagen",  # 5 chars - matches Copenhagen (but "Hagen" is a valid German city!)
    "park",  # 4 chars - matches Overland Park
    "ulm",  # 3 chars - matches Neu-Ulm (but "Ulm" is a valid German city!)
    "regen",  # 5 chars - matches Regensburg (but "Regen" is a valid German city!)
    "goch",  # 4 chars - matches Gochsheim (but "Goch" is a valid German city!)
}
