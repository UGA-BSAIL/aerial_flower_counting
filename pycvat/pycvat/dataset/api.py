"""
Tools for communicating with the CVAT API.
"""


from typing import Any

from swagger_client import ApiClient, Configuration


def make_api_client(
    *,
    username: str,
    password: str,
    host: str = "http://localhost:8080/api/v1",
    **kwargs: Any
) -> ApiClient:
    """
    Shortcut for creating a new ApiClient.

    Args:
        username: The username to log into CVAT with.
        password: The associated password for that user.
        host: The CVAT host to connect to.
        **kwargs: Will be applied directly to the underlying `Configuration`
            object.

    Returns:
        The `ApiClient` that it created.

    """
    config = Configuration()
    config.username = username
    config.password = password
    config.host = host
    # Some client-side validation features are broken in CVAT.
    config.client_side_validation = False

    # Apply keyword arguments.
    for key, value in kwargs.items():
        setattr(config, key, value)

    return ApiClient(config)
