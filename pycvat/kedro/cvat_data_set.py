"""
A Kedro `DataSet` for data from CVAT.
"""


from contextlib import ExitStack
from typing import Any, Dict, Tuple

from backports.cached_property import cached_property
from kedro.io import AbstractDataSet
from loguru import logger

from ..dataset.api import make_api_client
from ..dataset.task import Task


class CvatDataSet(AbstractDataSet):
    """
    A Kedro `DataSet` for data from CVAT.
    """

    def __init__(
        self,
        *,
        task_id: int,
        credentials: Dict[str, str],
        host: str = "http://localhost:8080/api/v1",
        **kwargs: Any
    ):
        """
        Args:
            task_id: The numerical ID of the task to load data from.
            credentials: Credentials to use for logging into CVAT. Should
                contain at least two keys: "username", which is the name of the
                username to log in as, and "password", which is the password for
                that username.
            host: The address of the CVAT server to connect to.
            **kwargs: Will be passed directly to the Swagger `Configuration`
                structure.
        """
        assert (
            "username" in credentials
        ), "'username' must be specified in CVAT credentials."
        assert (
            "password" in credentials
        ), "'password' must be specified in CVAT credentials."

        username = credentials["username"]
        password = credentials["password"]
        self.__api = make_api_client(
            username=username, password=password, host=host, **kwargs
        )

        self.__task_id = task_id

        # CVAT data is lazy-loaded, so this specifies whether the connection
        # was ever opened.
        self.__connected_to_cvat = False

    @cached_property
    def __init_cvat_handle(self) -> Tuple[Task, ExitStack]:
        """
        Initializes the CVAT handle that will be used to access this data.

        Returns:
            A handle object that can be used to access the CVAT task.

            Also, an `ExitStack` object that encapsulates callbacks for
            cleaning up the CVAT handle context. The `close()` method should
            be invoked manually on this object when the CVAT handle is no
            longer needed.

        """
        logger.info(
            "Initializing connection to task {}.", self.__task_id,
        )

        self.__connected_to_cvat = True

        with ExitStack() as exit_stack:
            handle = exit_stack.enter_context(
                Task.init_and_upload(
                    task_id=self.__task_id, api_client=self.__api
                )
            )

            context = exit_stack.pop_all()
            return handle, context

    @property
    def __cvat_task(self) -> Task:
        """
        Returns:
            The CVAT task handle to use. Will be created if it doesn't already
            exist.
        """
        task, _ = self.__init_cvat_handle
        return task

    @property
    def __cvat_context(self) -> ExitStack:
        """
        Returns:
            The `ExitStack` object that encapsulates callbacks for safely
            cleaning up the CVAT task handle.
        """
        _, context = self.__init_cvat_handle
        return context

    def __del__(self) -> None:
        # Clean up the context for the CVAT handle.
        if self.__connected_to_cvat:
            logger.debug("Cleaning up CVAT handle.")
            self.__cvat_context.close()
            self.__connected_to_cvat = False

    def _load(self) -> Task:
        return self.__cvat_task

    def _save(self, data: Task) -> None:
        # Force the data to be uploaded now.
        data.upload()

    def _exists(self) -> bool:
        # This is always true, because this class only works with data that
        # already exists on CVAT.
        return True

    # Not tested, because Kedro doesn't provide a public API for this.
    def _describe(self) -> Dict[str, Any]:  # pragma: no cover
        return dict(task_id=self.__task_id, host=self.__api.configuration.host)
