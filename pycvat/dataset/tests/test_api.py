"""
Tests for the `api` module.
"""


from faker import Faker
from pytest_mock import MockFixture

from pycvat.dataset import api


def test_make_api_client(mocker: MockFixture, faker: Faker) -> None:
    """
    Tests that `make_api_client` works.

    Args:
        mocker: The fixture to use for mocking.
        faker: The fixture to use for generating fake data.

    """
    # Arrange.
    # Mock out the dependencies.
    mock_config_class = mocker.patch(api.__name__ + ".Configuration")
    mock_api_client_class = mocker.patch(api.__name__ + ".ApiClient")

    # Create some fake credentials.
    username = faker.simple_profile()["username"]
    password = faker.pystr()
    host = faker.url()
    other = faker.pystr()

    # Act.
    got_client = api.make_api_client(
        username=username, password=password, host=host, other=other
    )

    # Assert.
    # Check the configuration.
    mock_config_class.assert_called_once_with()
    mock_config = mock_config_class.return_value
    assert mock_config.username == username
    assert mock_config.password == password
    assert mock_config.host == host
    assert mock_config.other == other

    # It should have created the API client.
    mock_api_client_class.assert_called_once_with(mock_config)
    assert got_client == mock_api_client_class.return_value
