
from pathlib import Path
from src.common.config import load_config


def test_config_loads():
    config = load_config(Path("config/config.yaml"))
    assert config["project"]["name"] == "musicstyle-cyclegan"
    assert config["model"]["input_shape"] == [128, 128, 1]
    assert config["audio"]["segment_seconds"] == 30
    assert config["training"]["batch_size"] >= 1
