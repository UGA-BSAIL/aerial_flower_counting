"""
Extensions to the Swagger-generated code.
"""


from pathlib import Path
from typing import Any

import requests
from requests.auth import HTTPBasicAuth
from swagger_client import Data, TasksApi


class ExtendedTasksApi(TasksApi):
    """
    Custom extensions for the generated `TasksApi` class.
    """

    def tasks_data_create(self, body: Data, id: int, **kwargs: Any) -> Data:
        """
        Swagger 2.0 does not support uploading an arbitrary number of files.
        This makes the generated version of `tasks_data_create` useless for
        uploading image data. This version of it fixes that problem manually.
        The API remains the same.

        Args:
            body: The data to upload in the request body.
            id: The task ID to upload data for.
            **kwargs: Additional arguments for Swagger.

        Returns:
            The response from the API, as a `Data` object.

        """
        # Client files should be passed in the files argument in order for
        # them to be uploaded correctly. It assumes that the files have been
        # set to a path on the local system.
        files = {}
        for i, client_file in enumerate(body.client_files):
            files[f"client_files[{i}]"] = Path(client_file.file).open("rb")
        form_data = body.to_dict()
        # We pass the client files separately.
        form_data["client_files"] = None

        config = self.api_client.configuration
        auth = HTTPBasicAuth(config.username, config.password)
        endpoint = f"{config.host}/tasks/{id}/data"
        response = requests.post(
            endpoint,
            auth=auth,
            data=form_data,
            files=files,
            verify=config.verify_ssl,
        )
        response.raise_for_status()

        return Data(**response.json())
