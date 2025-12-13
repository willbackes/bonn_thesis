"""All the general configuration of the project."""

from pathlib import Path

SRC = Path(__file__).parent.resolve()
RAW_DATA = SRC.joinpath("data", "raw").resolve()
ROOT = SRC.joinpath("..", "..").resolve()

BLD = ROOT.joinpath("bld").resolve()
DOCUMENTS = ROOT.joinpath("documents").resolve()
BUNDESLAND_DATA_BLD = BLD.joinpath("data", "bundesland_data").resolve()
SOEP_DATA_BLD = BLD.joinpath("data", "soep_data").resolve()
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

# OPENAI CONFIGURATION
OPENAI_API_MODELS = {
    "gpt-5-nano": {
        "input_cost": 0.05,
        "output_cost": 0.40,
    },
    "gpt-4.1-mini": {
        "input_cost": 0.40,
        "output_cost": 1.60,
        "training_cost": 5.00,
        "fine_tuned_input_cost": 0.80,
        "fine_tuned_output_cost": 3.20,
    },
    "gpt-4.1-nano": {
        "input_cost": 0.10,
        "output_cost": 0.40,
        "training_cost": 1.50,
        "fine_tuned_input_cost": 0.20,
        "fine_tuned_output_cost": 0.80,
    },
    "gpt-4o-mini": {
        "input_cost": 0.15,
        "output_cost": 0.60,
        "training_cost": 3.00,
        "fine_tuned_input_cost": 0.30,
        "fine_tuned_output_cost": 1.20,
    },
}

DEFAULT_OPENAI_MODEL = "gpt-4.1-nano"
BATCH_API_DISCOUNT = 0.50
