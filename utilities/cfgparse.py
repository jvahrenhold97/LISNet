import yaml


def load_config(path):
    """
    Loads and securely parses a YAML configuration file.

    Args:
        path (str | pathlib.Path): The path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the parsed configuration data.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

def parse_config(path):
    """
    Retrieves and parses the configuration file from the specified path.

    Args:
        path (str | pathlib.Path): The path to the configuration file.

    Returns:
        dict: A dictionary containing the parsed configuration data.
    """
    cfg = load_config(path)
    return cfg