"""
Tools for communicating with the CVAT API.
"""


from swagger_client import ApiClient, Configuration


def make_api_client(
    *, user: str, password: str, host: str = "http://localhost:8080/api/v1",
) -> ApiClient:
    """
    Shortcut for creating a new ApiClient.

    Args:
        user: The username to log into CVAT with.
        password: The associated password for that user.
        host: The CVAT host to connect to.

    Returns:
        The `ApiClient` that it created.

    """
    config = Configuration()
    config.username = user
    config.password = password
    config.host = host
    # Some client-side validation features are broken in CVAT.
    config.client_side_validation = False

    return ApiClient(config)
