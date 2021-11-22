"""
Tests for the `cvat_data_set` module.
"""

import unittest.mock as mock

import pytest
from faker import Faker
from pydantic.dataclasses import dataclass
from pytest_mock import MockFixture

from pycvat.dataset.task import Task
from pycvat.kedro import cvat_data_set
from pycvat.type_helpers import ArbitraryTypesConfig


class TestCvatDataSet:
    """
    Tests for the `CvatDataSet` class.
    """

    @dataclass(frozen=True, config=ArbitraryTypesConfig)
    class ConfigForTests:
        """
        Encapsulates standard configuration for most tests.

        Attributes:
            data_set: The `CvatDataSet` object under test.
            mock_task_class: The mocked `Task` class to use.
            mock_make_api_client: The mocked `make_api_client` function.

            task_id: The task ID to use for testing.
            username: The fake username for authentication.
            password: The fake password for authentication.
            host: The fake host for authentication.
        """

        data_set: cvat_data_set.CvatDataSet
        mock_task_class: mock.Mock
        mock_make_api_client: mock.Mock

        task_id: int
        username: str
        password: str
        host: str

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
        mock_task_class = mocker.patch(cvat_data_set.__name__ + ".Task")
        mock_make_api_client = mocker.patch(
            cvat_data_set.__name__ + ".make_api_client"
        )

        task_id = faker.random_int()
        username = faker.simple_profile()["username"]
        password = faker.pystr()
        host = faker.url()

        data_set = cvat_data_set.CvatDataSet(
            task_id=task_id,
            credentials={"username": username, "password": password},
            host=host,
        )

        yield cls.ConfigForTests(
            data_set=data_set,
            mock_task_class=mock_task_class,
            mock_make_api_client=mock_make_api_client,
            task_id=task_id,
            username=username,
            password=password,
            host=host,
        )

        # Manually call the destructor before the mocks exit scope, since it
        # needs the mocks to be active to work.
        data_set.__del__()

    def test_init(self, config: ConfigForTests) -> None:
        """
        Tests that the initialization process works correctly.

        Args:
            config: The configuration to use for testing.

        """
        # Assert.
        config.mock_make_api_client.assert_called_once_with(
            username=config.username,
            password=config.password,
            host=config.host,
        )

    @pytest.mark.parametrize(
        "cvat_connected",
        [False, True],
        ids=["cvat_not_connected", "cvat_connected"],
    )
    def test_del(self, config: ConfigForTests, cvat_connected: bool) -> None:
        """
        Tests that everything gets cleaned up properly.

        Args:
            config: The configuration to use for testing.
            cvat_connected: True if we should have actually connected to the
                CVAT server, false otherwise.

        """
        # Arrange.
        if cvat_connected:
            # We need to perform a load operation to force it to connect to
            # CVAT.
            config.data_set.load()

        # Act.
        config.data_set.__del__()

        # Assert.
        if cvat_connected:
            # It should have exited the context that it created.
            config.mock_task_class.init_and_upload.return_value.__exit__.assert_called_once()
        else:
            # It should not have touched the CVAT stuff.
            config.mock_task_class.init_and_upload.return_value.__exit__.assert_not_called()

    def test_load(self, config: ConfigForTests) -> None:
        """
        Tests that `load()` works.

        Args:
            config: The configuration to use for testing.

        """
        # Act.
        got_data = config.data_set.load()

        # Assert.
        # It should have set up the Task.
        mock_api = config.mock_make_api_client.return_value
        config.mock_task_class.init_and_upload.assert_called_once_with(
            task_id=config.task_id, api_client=mock_api
        )

        # It should not have exited the context.
        config.mock_task_class.init_and_upload.return_value.__exit__.assert_not_called()

        # It should have just given us the task.
        mock_task = (
            config.mock_task_class.init_and_upload.return_value.__enter__.return_value
        )
        assert got_data == mock_task

    def test_save(self, config: ConfigForTests, mocker: MockFixture) -> None:
        """
        Tests that `save()` works.

        Args:
            config: The configuration to use for testing.
            mocker: The fixture to use for mocking.

        """
        # Arrange.
        # Create fake data to try saving.
        mock_task = mocker.create_autospec(Task, instance=True)

        # Act.
        config.data_set.save(mock_task)

        # Assert.
        # It should have uploaded the data to the server.
        mock_task.upload.assert_called_once_with()

    def test_exists(self, config: ConfigForTests) -> None:
        """
        Tests that `exists()` works.

        Args:
            config: The configuration to use for testing.

        """
        # Act and assert.
        # Currently, this always returns True.
        assert config.data_set.exists()
