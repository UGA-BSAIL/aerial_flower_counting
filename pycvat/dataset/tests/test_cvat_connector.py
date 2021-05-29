"""
Tests for the `cvat_connector` module.
"""


from typing import Any

import pytest
from pydantic.dataclasses import dataclass
from pytest_mock import MockFixture
from swagger_client import ApiClient

from pycvat.dataset import cvat_connector
from pycvat.type_helpers import ArbitraryTypesConfig


class _Connector(cvat_connector.CvatConnector):
    """
    A `CvatConnector` subclass that is expressly for testing.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        # Tracks the number of times that various methods got called.
        self.__reload_called = 0
        self.__upload_called = 0

    def reload(self) -> None:
        self.__reload_called += 1

    def upload(self) -> None:
        self.__upload_called += 1

    @property
    def reload_called(self) -> int:
        """
        Returns:
            Number of times `reload()` was called.

        """
        return self.__reload_called

    @property
    def upload_called(self) -> int:
        """
        Returns:
            Number of times `upload()` was called.

        """
        return self.__upload_called


class TestCvatConnector:
    """
    Tests for the `CvatConnector` superclass.
    """

    @dataclass(frozen=True, config=ArbitraryTypesConfig)
    class ConfigForTests:
        """
        Encapsulates standard configuration for most tests.

        Attributes:
            connector: The `CvatConnector` under test.
            mock_api_client: The mocked `ApiClient` to use.
        """

        connector: cvat_connector.CvatConnector
        mock_api_client: ApiClient

    @classmethod
    @pytest.fixture
    def config(cls, mocker: MockFixture) -> ConfigForTests:
        """
        Generates standard configuration for most tests.

        Args:
            mocker: The fixture to use for mocking.

        Returns:
            The configuration that it generated.

        """
        mock_api_client = mocker.create_autospec(ApiClient, instance=True)

        connector = _Connector(api_client=mock_api_client)

        return cls.ConfigForTests(
            connector=connector, mock_api_client=mock_api_client
        )

    def test_init_and_upload(self, config: ConfigForTests) -> None:
        """
        Tests that `init_and_upload` works.

        Args:
            config: The configuration to use for testing.

        """
        # Act.
        with _Connector.init_and_upload(
            api_client=config.mock_api_client
        ) as connector:
            # Copy this so we can query it outside of the context.
            saved_connector = connector

        # Assert.
        # It should have uploaded automatically after exiting from the
        # context manager.
        assert saved_connector.upload_called == 1

    def test_api(self, config: ConfigForTests) -> None:
        """
        Tests that the `api` property works.

        Args:
            config: The configuration to use for testing.

        """
        # Act and assert.
        assert config.connector.api == config.mock_api_client
