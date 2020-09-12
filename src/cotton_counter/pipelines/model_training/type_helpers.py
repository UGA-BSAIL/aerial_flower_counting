"""
Miscellaneous type aliases and definitions.
"""


from typing import Tuple

Vector2I = Tuple[int, int]
"""
A 2D vector of integers.
"""


class ArbitraryTypesConfig:
    """
    Pydantic configuration class that allows for arbitrary types.
    """

    arbitrary_types_allowed = True
