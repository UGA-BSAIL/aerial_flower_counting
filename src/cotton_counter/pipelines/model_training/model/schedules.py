"""
Implements custom learning rate schedules.
"""


from pathlib import Path
from typing import Any, Dict, Union

import tensorflow as tf
from tensorflow.keras.optimizers import schedules


class LoggingWrapper(schedules.LearningRateSchedule):
    """
    Logs the current decayed learning rate to Tensorboard. I find this useful
    for debugging purposes, especially for more complex decay types.
    """

    def __init__(
        self,
        schedule: schedules.LearningRateSchedule,
        *,
        log_dir: Union[str, Path]
    ):
        """
        Args:
            schedule: The `LearningRateSchedule` that we are logging data from.
            log_dir: The directory to write TensorBoard summaries to.
        """
        self.__schedule = schedule
        self.__log_dir = Path(log_dir)
        self.__writer = tf.summary.create_file_writer(Path(log_dir).as_posix())

    def __call__(self, step: int) -> float:
        learning_rate = self.__schedule(step)

        # Log the learning rate.
        with self.__writer.as_default():
            tf.summary.scalar(
                "learning_rate", learning_rate, step=tf.cast(step, tf.int64)
            )

        return learning_rate

    def get_config(self) -> Dict[str, Any]:
        # Serialize the wrapped schedule.
        wrapped_serialized = schedules.serialize(self.__schedule)
        return dict(
            wrapped=wrapped_serialized, log_dir=self.__log_dir.as_posix()
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LoggingWrapper":
        # Deserialize the wrapped schedule.
        wrapped = schedules.deserialize(config["wrapped"])
        return cls(wrapped, log_dir=config["log_dir"])
