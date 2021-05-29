"""
Custom `Faker` providers that we use for testing.
"""


import unittest.mock as mock
from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Tuple

from faker import Faker
from faker.providers import BaseProvider
from swagger_client import (
    ClientFile,
    FrameMeta,
    Job,
    Label,
    LabeledData,
    LabeledImage,
    LabeledShape,
    LabeledTrack,
    SimpleJob,
    TrackedShape,
)

from pycvat.type_helpers import LabeledObject


class DataModelProvider(BaseProvider):
    """
    Provides fake versions of Swagger data models.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.__faker = Faker()

    def __point(
        self, frame_shape: Tuple[int, ...] = (1080, 1920)
    ) -> Tuple[float, float]:
        """
        Creates a single point annotation.

        Args:
            frame_shape: The shape of the frame that the annotations should
                fall within, as a tuple of (h, w, ...).

        Returns:
            The point that it created.

        """
        # Choose the x and y coordinates.
        max_height, max_width = frame_shape[:2]
        x = self.__faker.pyfloat(min_value=0, max_value=max_width)
        y = self.__faker.pyfloat(min_value=0, max_value=max_height)

        return x, y

    def __points_annotation(
        self,
        max_num_points: int = 5,
        frame_shape: Tuple[int, ...] = (1080, 1920),
    ) -> List[float]:
        """
        Creates a single fake `Points` instance.

        Args:
            max_num_points: The maximum number of points to allow.
            frame_shape: The shape of the frame that the annotations should
                fall within, as a tuple of (h, w, ...).

        Returns:
            The list of point coordinates, in the form of [x1, y1, x2, y2, ...]

        """
        # Generate some reasonable point data.
        num_points = self.random_int(min=1, max=max_num_points)
        point_coords = []
        for _ in range(num_points):
            point_coords.extend(self.__point(frame_shape=frame_shape))

        return point_coords

    def __set_common_label_attributes(
        self,
        labeled_object: LabeledObject,
        frame: Optional[int] = None,
        label_id: Optional[int] = None,
    ) -> None:
        """
        Sets the attributes common to all `LabeledObject`s.

        Args:
            labeled_object: The object to set the attributes on.
            frame: The frame number that is labeled. If not specified,
                it will be chosen randomly.
            label_id: The ID of the label to apply. If not specified, it will
                be chosen randomly.

        """
        if frame is None:
            frame = self.random_int()
        if label_id is None:
            label_id = self.random_int()

        labeled_object.id = self.random_int()
        labeled_object.frame = frame
        labeled_object.label_id = label_id
        labeled_object.group = self.random_int()
        labeled_object.source = self.__faker.text()
        # No attributes for now.
        labeled_object.attributes = []

    def simple_job(self, job_id: Optional[int] = None) -> SimpleJob:
        """
        Creates a fake `SimpleJob` object.

        Args:
            job_id: The ID for the job. If not specified, it will be chosen
                randomly.

        Returns:
            The `SimpleJob` that it created.

        """
        if job_id is None:
            job_id = self.random_int()

        job = mock.create_autospec(SimpleJob, instance=True)

        job.url = self.__faker.url()
        job.id = job_id
        job.assignee = self.__faker.name()
        job.status = self.random_element(
            ("annotation", "validation", "completed")
        )

        return job

    def job(
        self,
        start_frame: Optional[int] = None,
        stop_frame: Optional[int] = None,
        task_id: Optional[int] = None,
        **kwargs: Any
    ) -> Job:
        """
        Creates a fake `Job` object.

        Args:
            start_frame: The starting frame to use for the job. Will be
                chosen randomly if not specified.
            stop_frame: The ending frame to use for the job. Will be chosen
                randomly if not specified.
            task_id: The ID of the task that this job is associated with.
                Will be chosen randomly if not specified.
            **kwargs: Will be forwarded to `simple_job`.

        Returns:
            The `Job` that it created.

        """
        if start_frame is None:
            start_frame = self.random_int()
        if stop_frame is None:
            # Make sure it's after the start.
            stop_frame = start_frame + self.random_int()
        if task_id is None:
            task_id = self.random_int()

        # Create a SimpleJob and copy the attributes.
        simple_job = self.simple_job(**kwargs)

        job = mock.create_autospec(Job, instance=True)
        job.url = simple_job.url
        job.id = simple_job.id
        job.assignee = simple_job.assignee
        job.status = simple_job.status

        job.start_frame = start_frame
        job.stop_frame = stop_frame
        job.task_id = task_id

        return job

    def label(self, name: Optional[str] = None) -> Label:
        """
        Creates a fake `Label` object.

        Args:
            name: The name of the label to generate. If not specified,
                it will be randomly chosen.

        Returns:
            The `Label` that it created.

        """
        if name is None:
            name = self.__faker.word()

        mock_label = mock.create_autospec(Label, instance=True)
        mock_label.name = name

        return mock_label

    def frame_meta(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        name: Optional[str] = None,
    ) -> FrameMeta:
        """
        Creates a fake `FrameMeta` object.

        Args:
            width: The frame width to use. If not specified, will be
                chosen randomly.
            height: The frame height to use. If not specified, will be
                chosen randomly.
            name: The frame name to use. If not specified, will be chosen
                randomly.

        Returns:
            The `FrameMeta` that it created.

        """
        if width is None:
            width = self.random_int(max=4000)
        if height is None:
            height = self.random_int(max=4000)
        if name is None:
            name = self.__faker.file_name(category="image")

        mock_frame_meta = mock.create_autospec(FrameMeta, instance=True)
        mock_frame_meta.width = width
        mock_frame_meta.height = height
        mock_frame_meta.name = name

        return mock_frame_meta

    def labeled_image(self, **kwargs: Any) -> LabeledImage:
        """
        Creates a fake `LabeledImage` object.

        Args:
            **kwargs: Specify specific values for common label attributes.
                Anything that's not provided will be randomly chosen.

        Returns:
            The `LabeledImage` that it created.

        """
        labeled_image = mock.create_autospec(LabeledImage, instance=True)

        self.__set_common_label_attributes(labeled_image, **kwargs)

        return labeled_image

    def tracked_shape(
        self,
        points: Optional[List[float]] = None,
        frame: Optional[int] = None,
    ) -> TrackedShape:
        """
        Creates a fake `TrackedShape` object.

        Args:
            points: The point coordinates to use. If not specified, they will
                be chosen randomly.
            frame: The frame number that this shape is associated with. If
                not specified, it will be chosen randomly.

        Returns:
            The `TrackedShape` that it created.

        """
        if points is None:
            points = self.__points_annotation()
        if frame is None:
            frame = self.random_int()

        tracked_shape = mock.create_autospec(TrackedShape, instance=True)

        tracked_shape.type = self.random_element(
            ("rectangle", "polygon", "polyline", "points", "cuboid")
        )
        tracked_shape.occluded = self.__faker.pybool()
        tracked_shape.z_order = self.random_int()
        tracked_shape.points = points
        tracked_shape.id = self.random_int()
        tracked_shape.frame = frame
        tracked_shape.outside = self.__faker.pybool()
        # No attributes for now.
        tracked_shape.attributes = []

        return tracked_shape

    def labeled_shape(
        self, points: Optional[List[float]] = None, **kwargs: Any
    ) -> LabeledShape:
        """
        Creates a fake `LabeledShape` object.

        Args:
            points: The point coordinates to use. If not specified, they will
                be chosen randomly.
            **kwargs: Specify specific values for common label attributes.
                Anything that's not provided will be randomly chosen.

        Returns:
            The `LabeledShape` that it created.

        """
        if points is None:
            points = self.__points_annotation()

        labeled_shape = mock.create_autospec(LabeledShape, instance=True)

        labeled_shape.type = self.random_element(
            ("rectangle", "polygon", "polyline", "points", "cuboid")
        )
        labeled_shape.occluded = self.__faker.pybool()
        labeled_shape.z_order = self.random_int()
        labeled_shape.points = points
        self.__set_common_label_attributes(labeled_shape, **kwargs)

        return labeled_shape

    def labeled_track(
        self, max_num_shapes: int = 20, **kwargs: Any
    ) -> LabeledTrack:
        """
        Creates a fake `LabeledTrack` object.

        Args:
            max_num_shapes: The maximum number of tracked shapes to create.
            **kwargs: Specify specific values for common label attributes.
                Anything that's not provided will be randomly chosen.

        Returns:
            The `LabeledTrack` that it created.

        """
        labeled_track = mock.create_autospec(LabeledTrack, instance=True)

        # Create tracked shapes.
        num_shapes = self.random_int(max=max_num_shapes)
        labeled_track.shapes = [
            self.tracked_shape() for _ in range(num_shapes)
        ]

        self.__set_common_label_attributes(labeled_track, **kwargs)

        return labeled_track

    def labeled_data(
        self,
        min_frame: int = 0,
        max_frame: int = 100,
        max_num_tags: int = 5,
        min_num_tags: int = 0,
        max_num_shapes: int = 5,
        min_num_shapes: int = 0,
        max_num_tracks: int = 5,
        min_num_tracks: int = 0,
    ) -> LabeledData:
        """
        Creates a fake `LabeledData` object.

        Args:
            min_frame: The minimum frame number to use for annotations.
            max_frame: The maximum frame number to use for annotations.
            max_num_tags: Maximum number of tag annotations to generate.
            min_num_tags: Minimum number of tag annotations to generate.
            max_num_shapes: Maximum number of shape annotations to generate.
            min_num_shapes: Minimum number of shape annotations to generate.
            max_num_tracks: Maximum number of track annotations to generate.
            min_num_tracks: Minimum number of track annotations to generate.

        Returns:
            The `LabeledData` that it created.

        """
        num_tags = self.random_int(min=min_num_tags, max=max_num_tags)
        num_shapes = self.random_int(min=min_num_shapes, max=max_num_shapes)
        num_tracks = self.random_int(min=min_num_tracks, max=max_num_tracks)

        labeled_data = mock.create_autospec(LabeledData, instance=True)

        # Function for choosing a frame number.
        choose_frame = partial(self.random_int, min=min_frame, max=max_frame)

        labeled_data.version = self.random_int()
        labeled_data.tags = [
            self.labeled_image(frame=choose_frame()) for _ in range(num_tags)
        ]
        labeled_data.shapes = [
            self.labeled_shape(frame=choose_frame()) for _ in range(num_shapes)
        ]
        labeled_data.tracks = [
            self.labeled_track(frame=choose_frame()) for _ in range(num_tracks)
        ]

        return labeled_data

    def client_file(
        self, base_dir: Optional[Path] = None, create: bool = False
    ) -> ClientFile:
        """
        Creates a fake `ClientFile` object.

        Args:
            base_dir: Specify the base directory that all files will be a
                direct child of. If not specified, it will be chosen randomly.
            create: If true, actually create a blank file on the disk. Note
                that this should generally be used with `base_dir`, in order
                to make sure that the path it chooses is valid.

        Returns:
            The `ClientFile` it created.

        """
        client_file = mock.create_autospec(ClientFile, instance=True)

        if base_dir is not None:
            # Create in the specified directory.
            file_name = self.__faker.file_name(category="image")
            client_file.file = (base_dir / file_name).as_posix()
        else:
            client_file.file = self.__faker.file_path(category="image")

        if create:
            # Create a blank file.
            with Path(client_file.file).open("w"):
                pass

        return client_file
