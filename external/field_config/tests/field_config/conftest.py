from typing import Dict, Any

import pytest
import yaml

from src.field_config import FieldConfig
from .data import EXAMPLE_CONFIG_1


@pytest.fixture
def example_field_1_yaml() -> Dict[str, Any]:
    """
    Loads the raw YAML for the first example field configuration.

    Returns:
        The YAML data.

    """
    with EXAMPLE_CONFIG_1.open() as config_file:
        return yaml.safe_load(config_file)

@pytest.fixture
def example_field_1(example_field_1_yaml: Dict[str, Any]) -> FieldConfig:
    """
    Loads the first example field configuration.

    Args:
        example_field_1_yaml: The raw YAML data.

    Returns:
        The field configuration.

    """
    return FieldConfig.from_yml(example_field_1_yaml)