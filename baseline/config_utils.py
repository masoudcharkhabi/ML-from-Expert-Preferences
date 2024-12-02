import json
import logging


def load_config(config_file: str) -> dict:
    """
    Load configuration from a JSON file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: Configuration data loaded from the file.
    """
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
            validate_config(config)
            return config
    except FileNotFoundError:
        logging.error(f"Configuration file {config_file} not found.")
        exit(1)
    except json.JSONDecodeError:
        logging.error(f"Configuration file {config_file} is not a valid JSON.")
        exit(1)


def validate_config(config: dict):
    """
    Validate the configuration parameters.

    Args:
        config (dict): Configuration data.
    """
    required_keys = ["bbh_dir", "flan_file", "BBH_MAX_TOKEN_LENGTH", "FLAN_MAX_NEW_TOKENS", "MAX_EXAMPLES", "model_name"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    if not isinstance(config["BBH_MAX_TOKEN_LENGTH"], int) or config["BBH_MAX_TOKEN_LENGTH"] <= 0:
        raise ValueError("BBH_MAX_TOKEN_LENGTH must be a positive integer")
