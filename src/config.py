import yaml
import json
import jsonschema
from pydantic import BaseModel
from enum import Enum
from constants import SCHEMA_BASE_PATH, CONFIG_PATH_ENV_VAR
import dotenv
import os



_cache = None


class MCTS(BaseModel):
    iterations: int
    workers: int

class Device(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"

class Opponent(Enum):
    SELF = "self"
    STOCKFISH = "stockfish"
    USER = "user"
    MIXED = "mixed"

class Model(BaseModel):
    path: str
    epochs: int
    device: Device
    learning_rate: float
    mcts: MCTS
    opponent: Opponent


class Stockfish(BaseModel):
    path: str
    depth: int
    nodes: int
    time: float


class Config(BaseModel):
    log_path: str
    model: Model
    stockfish: Stockfish


def validate_config(config: dict):
    with open(SCHEMA_BASE_PATH + "config_schema.json", "r") as f:
        schema = json.load(f)
    
    jsonschema.validate(config, schema)


def load_config(path: str = None) -> Config:
    if path is None:
        dotenv.load_dotenv()
        path = os.getenv(CONFIG_PATH_ENV_VAR) or "config.yaml"

    global _cache
    if _cache is not None:
        return _cache
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        validate_config(config)
        _cache = Config(**config)
        return _cache
