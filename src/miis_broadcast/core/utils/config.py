from pathlib import Path
from typing import Any, Mapping

import yaml


def parse_configs(config_path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)
    return configs


def deep_merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Return a new dict: keys in override win; nested dicts are merged recursively."""
    out: dict = dict(base)
    for k, v in override.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, Mapping)
        ):
            out[k] = deep_merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def load_app_config_with_local(
    app_yml_path: str | Path,
    *,
    local_yml_name: str = "app.local.yml",
) -> dict:
    """Load app.yml, then merge configs/app.local.yml if it exists (gitignored).

    Use app.local.yml for API keys and other secrets so they are not committed.
    """
    path = Path(app_yml_path)
    configs = parse_configs(path)
    local_path = path.parent / local_yml_name
    if local_path.is_file():
        local_cfg = parse_configs(local_path)
        if isinstance(local_cfg, dict):
            configs = deep_merge_dict(configs, local_cfg)
    return configs