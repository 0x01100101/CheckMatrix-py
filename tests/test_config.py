import pytest
from config import Config, Device, load_config



@pytest.fixture
def config():
    return load_config()


# Ensure config values are default
def test_config(config: Config):
    assert config.log_path == "logs/checkmatrix.log"

    assert config.model.path == "data/model.pth"
    assert config.model.epochs == -1
    assert config.model.device == Device.AUTO
    assert config.model.learning_rate == 0.001
    assert config.model.mcts.iterations == 300
    assert config.model.mcts.workers == 6
    assert config.model.opponent == "self"

    assert config.stockfish.path == "stockfish.exe"
    assert config.stockfish.depth == 10
    assert config.stockfish.nodes == 100000
    assert config.stockfish.time == 0.1
