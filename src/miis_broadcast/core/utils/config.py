import yaml

def parse_configs(config_path) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        configs = yaml.safe_load(f)
    return configs