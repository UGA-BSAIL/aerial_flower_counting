"""
Miscellaneous type aliases and definitions.
"""


from typing import Collection, Dict, Tuple, Union

import tensorflow as tf

Vector2I = Tuple[int, int]
"""
A 2D vector of integers.
"""
DatasetOutput = Union[tf.Tensor, Collection[tf.Tensor], Dict[str, tf.Tensor]]
"""
Represents the type of objects that can be produced by `tf.data.Dataset`.
"""


class ArbitraryTypesConfig:
    """
    Pydantic configuration class that allows for arbitrary types.
    """

    arbitrary_types_allowed = True
