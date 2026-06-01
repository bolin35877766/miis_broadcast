import yaml

def parse_configs(config_path) -> dict:
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    return configs