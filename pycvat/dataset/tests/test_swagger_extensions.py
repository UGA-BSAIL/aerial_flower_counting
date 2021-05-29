"""
Tests for the `swagger_extensions` module.
"""


import unittest.mock as mock
from functools import partial
from pathlib import Path

import pytest
from faker import Faker
from pydantic.dataclasses import dataclass
from pytest_mock import MockFixture
from swagger_client import ApiClient, Configuration, Data

from pycvat.dataset import swagger_extensions
from pycvat.type_helpers import ArbitraryTypesConfig


class TestExtendedTasksApi:
    """
    Tests for the `ExtendedTasksApi` class.
    """

    @dataclass(frozen=True, config=ArbitraryTypesConfig)
    class ConfigForTests:
        """
        Encapsulates standard configuration for most tests.

        Attributes:
            api: The `ExtendedTasksApi` object under test.
            mock_post: The mocked `requests.post` function.
            mock_data_class: The mocked `Data` class.
            mock_api_client: The mocked `ApiClient` class to test with.
            mock_basic_auth_class: The mocked `HTTPBasicAuth` class.
        """

        api: swagger_extensions.ExtendedTasksApi
        mock_post: mock.Mock
        mock_data_class: mock.Mock
        mock_api_client: ApiClient
        mock_basic_auth_class: mock.Mock

    @classmethod
    @pytest.fixture
    def config(cls, mocker: MockFixture, faker: Faker) -> ConfigForTests:
        """
        Generates standard configuration for most tests.

        Args:
            mocker: The fixture to use for mocking.
            faker: The fixture to use for generating fake data.

        Returns:
            The configuration that it generated.

        """
        # Mock the dependencies.
        mock_post = mocker.patch("requests.post")
        mock_data_class = mocker.patch(swagger_extensions.__name__ + ".Data")
        mock_api_client = mocker.create_autospec(ApiClient, instance=True)
        mock_basic_auth_class = mocker.patch(
            swagger_extensions.__name__ + ".HTTPBasicAuth"
        )

        # Fake the configuration data.
        mock_api_client.configuration = mocker.create_autospec(
            Configuration, instance=True
        )
        mock_api_client.configuration.username = faker.pystr()
        mock_api_client.configuration.password = faker.pystr()
        mock_api_client.configuration.host = faker.url()

        api = swagger_extensions.ExtendedTasksApi(mock_api_client)

        return cls.ConfigForTests(
            api=api,
            mock_post=mock_post,
            mock_data_class=mock_data_class,
            mock_api_client=mock_api_client,
            mock_basic_auth_class=mock_basic_auth_class,
        )

    def test_tasks_data_create(
        self,
        config: ConfigForTests,
        mocker: MockFixture,
        faker: Faker,
        tmp_path: Path,
    ) -> None:
        """
        Tests that `tasks_data_create` works.

        Args:
            config: The configuration to use for testing.
            mocker: The fixture to use for mocking.
            faker: The fixture to use for generating fake data.
            tmp_path: Directory to use for creating temporary files.

        """
        # Arrange.
        # Create some fake data to upload.
        mock_data = mocker.create_autospec(Data, instance=True)
        make_file = partial(faker.client_file, base_dir=tmp_path, create=True)
        mock_data.client_files = [make_file(), make_file()]

        task_id = faker.random_int()

        # Act.
        uploaded = config.api.tasks_data_create(mock_data, task_id)

        # Assert.
        # It should have authenticated correctly.
        mock_config = config.mock_api_client.configuration
        config.mock_basic_auth_class.assert_called_once_with(
            mock_config.username, mock_config.password
        )
        mock_auth = config.mock_basic_auth_class.return_value

        # It should have made the request.
        config.mock_post.assert_called_once_with(
            mock.ANY,
            auth=mock_auth,
            data=mock_data.to_dict.return_value,
            files=mock.ANY,
        )
        args, kwargs = config.mock_post.call_args

        # It should have included the correct files.
        assert len(kwargs["files"]) == len(mock_data.client_files)

        # It should have posted to the correct endpoint.
        endpoint = args[0]
        assert mock_config.host in endpoint

        # It should have produced a new Data instance.
        config.mock_data_class.assert_called_once()
        assert uploaded == config.mock_data_class.return_value
