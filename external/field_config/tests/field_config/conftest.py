import pytest
import yaml

from src.field_config import FieldConfig
from .data import EXAMPLE_CONFIG_1


@pytest.fixture
def example_field_1() -> FieldConfig:
    """
    Loads the first example field configuration.

    Returns:
        The field configuration.

    """
    with EXAMPLE_CONFIG_1.open() as config_file:
        config = yaml.safe_load(config_file)
    return FieldConfig.from_yml(config)