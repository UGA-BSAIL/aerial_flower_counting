import itertools
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger
from singledispatchmethod import singledispatchmethod
from swagger_client import Job as JobModel
from swagger_client import (
    JobsApi,
    LabeledData,
    LabeledImage,
    LabeledShape,
    LabeledTrack,
)

from ..type_helpers import LabeledObject
from .clearable_cached_property import ClearableCachedProperty
from .cvat_connector import CvatConnector


class Job(CvatConnector):
    """
    Represents a particular job on the CVAT server.
    """

    def __init__(self, *, job_id: int, **kwargs: Any):
        """
        Args:
            job_id: The numerical ID of the job.
            **kwargs: Will be forwarded to the superclass.
        """
        super().__init__(**kwargs)

        self.__job_id = job_id
        self.__jobs_api = JobsApi(self.api)

    @ClearableCachedProperty
    def __job_data(self) -> JobModel:
        """
        Returns:
            Basic data about this job.

        """
        logger.debug("Downloading basic info for job {}.", self.__job_id)
        return self.__jobs_api.jobs_read(self.__job_id)

    @ClearableCachedProperty
    def __annotations(self) -> LabeledData:
        """
        Gets all the annotations associated with this job.

        Returns:
            The raw annotation data.

        """
        logger.debug("Downloading annotations for job {}.", self.__job_id)
        return self.__jobs_api.jobs_annotations_read(self.__job_id)

    @classmethod
    def __append_or_create(
        cls, my_dict: Dict[Any, List], key: Any, value: Any
    ) -> None:
        """
        Utility for dealing with dicts that have lists as elements. If a key
        already exists, it appends the element. Otherwise, it adds the key
        and sets the value to a new list with that element as its single
        member.

        Args:
            my_dict: Dictionary to add to.
            key: The key to add.
            value: The value to add.

        """
        if key not in my_dict:
            my_dict[key] = []
        my_dict[key].append(value)

    @classmethod
    def __frames_to_annotations(
        cls, annotations: Iterable[LabeledObject]
    ) -> Dict[int, List[LabeledObject]]:
        """
        Takes raw annotations and creates a dict that maps frame numbers to
        all the annotations for that frame.

        Args:
            annotations: The raw annotations.

        Returns:
            The mapping.

        """
        frames_to_annotations = {}
        for annotation in annotations:
            cls.__append_or_create(
                frames_to_annotations, annotation.frame, annotation
            )

        return frames_to_annotations

    @ClearableCachedProperty
    def __frames_to_tags(self) -> Dict[int, List[LabeledImage]]:
        """
        Returns:
            A mapping of frame numbers to corresponding tag annotations.

        """
        return self.__frames_to_annotations(self.__annotations.tags)

    @ClearableCachedProperty
    def __frames_to_shapes(self) -> Dict[int, LabeledShape]:
        """
        Returns:
            A mapping of frame numbers to corresponding shape annotations.

        """
        return self.__frames_to_annotations(self.__annotations.shapes)

    @ClearableCachedProperty
    def __frames_to_tracks(self) -> Dict[int, LabeledTrack]:
        """
        Returns:
            A mapping of frame numbers to corresponding track annotations.

        """
        return self.__frames_to_annotations(self.__annotations.tracks)

    def annotations_for_frame(self, frame_num: int) -> List[LabeledObject]:
        """
        Gets all the annotations for a particular frame.

        Args:
            frame_num: The frame number to get annotations for.

        Returns:
            All of the annotations for that frame.

        """
        annotations = []

        annotations.extend(self.__frames_to_tags.get(frame_num, []))
        annotations.extend(self.__frames_to_shapes.get(frame_num, []))
        annotations.extend(self.__frames_to_tracks.get(frame_num, []))

        return annotations

    def iter_annotations(
        self, start_at: Optional[int] = None
    ) -> Iterable[List[LabeledObject]]:
        """
        Iterates through all the annotations for this job, in order.

        Args:
            start_at: The frame number to start at. If not specified, it will
                start at the first frame in this job.

        Yields:
            The annotations for each frame.

        """
        if start_at is None:
            # Start at the first frame.
            start_at = self.start_frame

        for frame_num in range(start_at, self.end_frame):
            yield self.annotations_for_frame(frame_num)

    @singledispatchmethod
    def update_annotations(self, annotations: Any) -> None:  # pragma: no cover
        """
        Updates the annotations for a single frame.

        Args:
            annotations: The additional annotations for that frame. The frame
                number will be read from the annotations object.

        """
        raise NotImplementedError(
            f"update_annotations() does not support annotations of type"
            f" {type(annotations)}."
        )

    @update_annotations.register
    def _(self, annotations: LabeledImage) -> None:
        self.__append_or_create(
            self.__frames_to_tags, annotations.frame, annotations
        )

    @update_annotations.register
    def _(self, annotations: LabeledShape) -> None:
        self.__append_or_create(
            self.__frames_to_shapes, annotations.frame, annotations
        )

    @update_annotations.register
    def _(self, annotations: LabeledTrack) -> None:
        self.__append_or_create(
            self.__frames_to_tracks, annotations.frame, annotations
        )

    def upload(self) -> None:
        logger.debug("Uploading annotations to CVAT.")

        # Flatten all the annotations.
        tags = list(
            itertools.chain.from_iterable(self.__frames_to_tags.values())
        )
        shapes = list(
            itertools.chain.from_iterable(self.__frames_to_shapes.values())
        )
        tracks = list(
            itertools.chain.from_iterable(self.__frames_to_tracks.values())
        )

        # Generate the new annotation data.
        annotations = LabeledData(
            version=self.__annotations.version,
            tags=tags,
            shapes=shapes,
            tracks=tracks,
        )

        self.__jobs_api.jobs_annotations_update(annotations, self.__job_id)

    def reload(self) -> None:
        logger.debug("Forcing data reload.")

        Job.__job_data.flush_cache(self)
        Job.__annotations.flush_cache(self)
        Job.__frames_to_tags.flush_cache(self)
        Job.__frames_to_shapes.flush_cache(self)
        Job.__frames_to_tracks.flush_cache(self)

    @property
    def start_frame(self) -> int:
        """
        Returns:
            The frame that this job starts at.

        """
        return int(self.__job_data.start_frame)

    @property
    def end_frame(self) -> int:
        """
        Returns:
            The frame that this job ends at.

        """
        return int(self.__job_data.stop_frame)
