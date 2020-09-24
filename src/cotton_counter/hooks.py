"""
Kedro hooks for implementing custom behavior.
"""


from typing import Iterable

from kedro.config import ConfigLoader, TemplatedConfigLoader
from kedro.framework.hooks import hook_impl


class ProjectHooks:
    """
    Hooks specific to this project.
    """

    @hook_impl
    def register_config_loader(
        self, conf_paths: Iterable[str]
    ) -> ConfigLoader:
        return TemplatedConfigLoader(
            conf_paths,
            # Read the dictionary of global variables from the project config.
            globals_pattern="*globals.yml",
        )
