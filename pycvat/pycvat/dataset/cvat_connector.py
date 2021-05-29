"""
Superclass for objects that connect to CVAT.
"""


import abc
from contextlib import contextmanager
from typing import Any, ContextManager

from swagger_client import ApiClient


class CvatConnector(abc.ABC):
    """
    Superclass for objects that connect to CVAT.
    """

    @classmethod
    @contextmanager
    def init_and_upload(
        cls, *args: Any, **kwargs: Any
    ) -> ContextManager["CvatConnector"]:
        """
        Initializes a new object as a context manager. It will automatically
        call `upload()` after the context is exited.

        Args:
            *args: Will be forwarded to the class constructor.
            **kwargs: Will be forwarded to the class constructor.

        Returns:
            The context manager that it created.

        """
        connector = cls(*args, **kwargs)
        yield connector

        # Upload after we're finished.
        connector.upload()

    def __init__(self, *, api_client: ApiClient):
        """
        Args:
            api_client: The client to use for connecting to the CVAT API.
        """
        self.__jobs_api = api_client

    @property
    def api(self) -> ApiClient:
        """
        Returns:
            The `ApiClient` that this object was created with.

        """
        return self.__jobs_api

    @abc.abstractmethod
    def reload(self) -> None:
        """
        Forces all downloaded data from CVAT to be reloaded.

        """

    @abc.abstractmethod
    def upload(self) -> None:
        """
        Updates CVAT to match the current local state of this object.

        """
