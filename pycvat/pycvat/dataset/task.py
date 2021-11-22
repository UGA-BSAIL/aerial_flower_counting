"""
Manages downloading and opening annotations from a CVAT task.
"""


from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
from loguru import logger
from methodtools import lru_cache
from swagger_client import ApiClient, ClientFile, Data, DataMeta, Label
from swagger_client import Task as TaskModel
from swagger_client import TasksApi

from .clearable_cached_property import ClearableCachedProperty
from .cvat_connector import CvatConnector
from .job import Job
from .swagger_extensions import ExtendedTasksApi


class Task(CvatConnector):
    """
    Manages downloading and opening annotations from a CVAT task.
    """

    _MAX_IMAGES_TO_CACHE = 256
    """
    Maximum number of downloaded images to cache in memory.
    """

    @classmethod
    def create_new(
        cls,
        *,
        api_client: ApiClient,
        name: str,
        labels: Iterable[Label],
        bug_tracker: str = "",
        images: List[Path] = [],
        image_quality: int = 70,
    ) -> "Task":
        """
        Creates a brand new task and uploads it to the server.

        Args:
            api_client: The client to use for accessing the CVAT API.
            name: The name of the new task.
            labels: The annotation labels that should be used for the new task.
            bug_tracker: Specify the URL of a bug tracker for the new task.
            images: A list of the paths to the images to annotate for this task.
            image_quality: Image quality to use for the task, between 0 and 100.

        Returns:
            The task that it created.

        """
        # Create the task on the server.
        api = ExtendedTasksApi(api_client)
        task_model = TaskModel(
            name=name, labels=labels, bug_tracker=bug_tracker
        )
        task_model = api.tasks_create(task_model)
        logger.info("Created new task with ID {}.", task_model.id)

        # Add the images to the task.
        logger.debug("Uploading task images...")
        client_files = [ClientFile(file=i) for i in images]
        task_data = Data(
            image_quality=image_quality, client_files=client_files
        )
        api.tasks_data_create(task_data, task_model.id)

        return cls(task_id=task_model.id, api_client=api_client)

    def __init__(self, *, task_id: int, **kwargs: Any):
        """
        Args:
            task_id: The numerical ID of the task.
            **kwargs: Will be forwarded to the superclass.
        """
        super().__init__(**kwargs)

        self.__task_api = TasksApi(self.api)
        self.__task_id = task_id

    @ClearableCachedProperty
    def __task_data(self) -> TaskModel:
        """
        Gets the general task data from the API.

        Returns:
            The task data that it got.

        """
        logger.debug("Downloading data for task {}.", self.__task_id)
        return self.__task_api.tasks_read(self.__task_id)

    @ClearableCachedProperty
    def __task_metadata(self) -> DataMeta:
        """
        Gets the task metadata from the API.

        Returns:
            The task metadata that it got.

        """
        logger.debug("Downloading metadata for task {}.", self.__task_id)
        return self.__task_api.tasks_data_data_info(self.__task_id)

    @ClearableCachedProperty
    def __image_name_to_frame_num(self) -> Dict[str, int]:
        """
        Returns:
            A mapping of image names to their corresponding frame numbers.

        """
        frame_data = self.__task_metadata.frames
        return {f.name: i for i, f in enumerate(frame_data)}

    def __download_image(self, frame_num: int) -> np.ndarray:
        """
        Downloads an image from the CVAT server.

        Args:
            frame_num: The number of the frame to load.

        Returns:
            The raw image data that it downloaded.

        """
        # Download the image.
        logger.debug("Downloading image for frame {}.", frame_num)
        task = self.__task_api.tasks_data_read(
            self.__task_id,
            "frame",
            "original",
            frame_num,
            # This is necessary, because otherwise Swagger tries to decode
            # the image data as UTF-8.
            _preload_content=False,
        )

        return np.frombuffer(task.data, dtype=np.uint8)

    @lru_cache(maxsize=_MAX_IMAGES_TO_CACHE)
    def get_image(
        self, frame_num: int, compressed: bool = False
    ) -> np.ndarray:
        """
        Loads a particular image from the CVAT server.

        Args:
            frame_num: The number of the frame to load.
            compressed: If true, it will return the raw JPEG data instead of
                loading it.

        Returns:
            The image that it loaded.

        """
        image = self.__download_image(frame_num)

        if not compressed:
            # Load the image data.
            return cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        else:
            return image

    def get_image_size(self, frame_num: int) -> Tuple[int, int]:
        """
        Gets the width and height of a frame in pixels.

        Args:
            frame_num: The frame number to get the size of.

        Returns:
            The width and height of the frame.

        """
        frame_info = self.__task_metadata.frames[frame_num]
        return frame_info.width, frame_info.height

    @ClearableCachedProperty
    def jobs(self) -> List[Job]:
        """
        Gets all the jobs associated with this task.

        Returns:
            The jobs for the task.

        """
        # Get the job numbers.
        segments = self.__task_data.segments
        all_jobs = []
        for segment in segments:
            all_jobs.extend([j.id for j in segment.jobs])
        logger.debug("Got job IDs {} for task {}.", all_jobs, self.__task_id)

        # Create the Job objects.
        return [Job(job_id=i, api_client=self.api) for i in all_jobs]

    def get_labels(self) -> List[Label]:
        """
        Returns:
            The labels associated with this task.

        """
        return self.__task_data.labels[:]

    def find_label(self, name: str) -> Label:
        """
        Gets a label with a specific name.

        Args:
            name: The name of the desired label.

        Returns:
            The label it found.

        Raises:
            NameError if there is no label with that name.

        """
        for label in self.get_labels():
            if label.name == name:
                return label

        raise NameError(f"There is no label with name '{name}'.")

    def find_image_frame_num(self, name: str) -> int:
        """
        Finds an image's frame number by its name.

        Args:
            name: The name of the frame to find.

        Returns:
            The frame number.

        """
        return self.__image_name_to_frame_num[name]

    @property
    def id(self) -> int:
        """
        Returns:
            The ID for the task.

        """
        return self.__task_id

    @property
    def num_frames(self) -> int:
        """
        Returns:
            The total number of frames associated with the task.

        """
        # Find the largest frame number in any segment.
        segments = self.__task_data.segments
        largest_frame = 0
        for segment in segments:
            largest_frame = max(largest_frame, segment.stop_frame)

        return largest_frame

    def reload(self) -> None:
        logger.debug("Forcing data reload.")

        Task.__task_data.flush_cache(self)
        Task.__task_metadata.flush_cache(self)
        Task.__image_name_to_frame_num.flush_cache(self)
        Task.jobs.flush_cache(self)

        # Clear all cached images.
        self.get_image.cache_clear()

    def upload(self) -> None:
        logger.debug("Uploading task data to CVAT.")

        # Make sure all the job data is up-to-date.
        for job in self.jobs:
            job.upload()
