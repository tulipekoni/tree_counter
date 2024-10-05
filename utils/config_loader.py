import json
import os
from typing import Dict, Any

def load_config(config_path: str) -> Dict[Any, Any]:
    """
    Load configuration from a JSON file.

    Args:
        config_path (str): Path to the configuration JSON file.

    Returns:
        Dict[Any, Any]: Dictionary containing the configuration.

    Raises:
        FileNotFoundError: If the config file is not found.
        json.JSONDecodeError: If the config file is not valid JSON.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in config file: {config_path}", e.doc, e.pos)

def override_config(config: Dict[Any, Any], overrides: Dict[str, str]) -> Dict[Any, Any]:
    """
    Override configuration parameters with command-line arguments.

    Args:
        config (Dict[Any, Any]): Original configuration dictionary.
        overrides (Dict[str, str]): Dictionary of override parameters.

    Returns:
        Dict[Any, Any]: Updated configuration dictionary.
    """
    for key, value in overrides.items():
        if key in config:
            # Convert value to appropriate type
            if isinstance(config[key], bool):
                config[key] = value.lower() in ('true', '1', 'yes')
            elif isinstance(config[key], int):
                config[key] = int(value)
            elif isinstance(config[key], float):
                config[key] = float(value)
            else:
                config[key] = value
    return config
