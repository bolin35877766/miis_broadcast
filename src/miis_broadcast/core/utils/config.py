from pathlib import Path
import yaml


def parse_configs(config_path) -> dict:
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    return configs


def get_configs_dir() -> Path:
    """Walk up from this file until a configs/ directory is found."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "configs"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("configs/ directory not found relative to config.py")


def load_app_config() -> dict:
    return parse_configs(get_configs_dir() / "app.yml")


def load_models_config() -> dict:
    return parse_configs(get_configs_dir() / "models.yml")


def load_system_prompts() -> dict:
    return parse_configs(get_configs_dir() / "system_prompts.yml")
