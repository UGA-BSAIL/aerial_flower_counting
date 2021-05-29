"""
Tests for the `tracker` module.
"""

import unittest.mock as mock

import cv2
import numpy as np
import pytest
import yaml
from faker import Faker
from pydantic.dataclasses import dataclass
from pytest_mock import MockFixture
from pytest_snapshot.plugin import Snapshot

from pycvat.dataset.job import Job
from pycvat.dataset.task import Task
from pycvat.tracking import tracker
from pycvat.type_helpers import ArbitraryTypesConfig

from .test_images import FRAME_1_PATH, FRAME_2_PATH


class TestTracker:
    """
    Tests for the `Tracker` class.
    """

    @dataclass(frozen=True, config=ArbitraryTypesConfig)
    class ConfigForTests:
        """
        Encapsulates standard configuration for most tests.

        Attributes:
            tracker: The `Tracker` instance under test.
            mock_task: The mocked `Task` to use with the tracker.
            mock_sift: The mocked OpenCV SIFT object.
            mock_matcher: The mocked OpenCV feature matcher.
            mock_labeled_shape_class: The mocked `LabeledShape` class.
            mock_pyr_down: The mocked `cv2.pyrDown` function.
            mock_perspective_transform: The mocked `cv2.perspectiveTransform`
                function.
            mock_find_homography: The mocked `cv2.findHomography` function.
        """

        tracker: tracker.Tracker
        mock_task: mock.Mock
        mock_sift: mock.Mock
        mock_matcher: mock.Mock
        mock_labeled_shape_class: mock.Mock
        mock_pyr_down: mock.Mock
        mock_perspective_transform: mock.Mock
        mock_find_homography: mock.Mock

    @classmethod
    @pytest.fixture
    def config(cls, mocker: MockFixture) -> ConfigForTests:
        """
        Generates standard configuration for tests.

        Args:
            mocker: The fixture to use for mocking.

        Returns:
            The configuration that it generated.

        """
        # Mock the dependencies.
        # create_autospec doesn't work write on methods decorated with
        # lru_cache.
        mock_task = mocker.Mock(spec=Task)
        mock_sift_create = mocker.patch("cv2.SIFT_create")
        mock_sift = mock_sift_create.return_value
        mock_matcher_create = mocker.patch("cv2.FlannBasedMatcher_create")
        mock_matcher = mock_matcher_create.return_value
        mock_labeled_shape_class = mocker.patch(
            tracker.__name__ + ".LabeledShape"
        )
        mock_pyr_down = mocker.patch("cv2.pyrDown")
        mock_perspective_transform = mocker.patch("cv2.perspectiveTransform")
        mock_find_homography = mocker.patch("cv2.findHomography")

        # Create the Tracker under test.
        _tracker = tracker.Tracker(mock_task)

        return cls.ConfigForTests(
            tracker=_tracker,
            mock_task=mock_task,
            mock_sift=mock_sift,
            mock_matcher=mock_matcher,
            mock_labeled_shape_class=mock_labeled_shape_class,
            mock_pyr_down=mock_pyr_down,
            mock_perspective_transform=mock_perspective_transform,
            mock_find_homography=mock_find_homography,
        )

    @pytest.mark.parametrize(
        "start_frame", [0, 5], ids=["start_frame_0", "start_frame_5"]
    )
    def test_track_forward(
        self,
        config: ConfigForTests,
        mocker: MockFixture,
        faker: Faker,
        start_frame: int,
    ) -> None:
        """
        Tests that `track_forward()` works.

        Args:
            config: The configuration to use for testing.
            mocker: The fixture to use for mocking.
            faker: The fixture to use for generating fake data.
            start_frame: The frame to start tracking from during the test.

        """
        # Arrange.
        # Create some fake annotations.
        first_annotations = faker.labeled_data(
            min_frame=start_frame, max_frame=start_frame
        ).shapes

        # Make it look like we have some frame data.
        mock_first_frame = mocker.create_autospec(np.ndarray, instance=True)
        mock_next_frame = mocker.create_autospec(np.ndarray, instance=True)
        config.mock_task.get_image.side_effect = [
            mock_first_frame,
            mock_next_frame,
        ]

        # create_autospec doesn't work on methods decorated with singledispatch.
        mock_job = mocker.Mock(spec=Job)
        mock_job.annotations_for_frame.side_effect = [
            first_annotations,
            mocker.DEFAULT,
        ]
        config.mock_task.jobs = [mock_job]

        # Make it look like SIFT feature extraction produces valid results.
        config.mock_sift.detectAndCompute.return_value = (
            mocker.Mock(),
            mocker.Mock(),
        )
        config.mock_matcher.return_value = [mocker.Mock(), mocker.Mock()]
        config.mock_find_homography.return_value = (
            mocker.Mock(),
            mocker.Mock(),
        )

        # Make sure that our fake perspective transformation produces valid
        # points. In this case, we're just going to have it add 1 to the
        # input points.
        config.mock_perspective_transform.side_effect = lambda p, _: p + 1

        # Act.
        updated_annotations = config.tracker.track_forward(
            start_frame=start_frame
        )

        # Assert.
        # It should have saved the additional annotations.
        mock_job.update_annotations.assert_has_calls(
            [mocker.call(config.mock_labeled_shape_class.return_value)]
            * len(first_annotations)
        )

        # It should have returned the combined annotations.
        mock_job.annotations_for_frame.assert_any_call(start_frame + 1)
        assert (
            updated_annotations == mock_job.annotations_for_frame.return_value
        )

    @pytest.mark.slow
    @pytest.mark.integration
    def test_track_forward_integration(
        self, snapshot: Snapshot, mocker: MockFixture, faker: Faker
    ) -> None:
        """
        Full integration test of the `track_forward()` method that uses a
        snapshot of the tracked annotations from real data.

        Args:
            snapshot: The fixture to use for snapshot testing.
            mocker: The fixture to use for mocking.
            faker: The fixture to use for generating fake data.

        """
        # Arrange.
        # Load some real frame data.
        frame_1 = cv2.imread(FRAME_1_PATH.as_posix())
        frame_2 = cv2.imread(FRAME_2_PATH.as_posix())

        # Create some fake annotations for those frames also.
        annotations = faker.labeled_data(max_frame=0).shapes

        # Mock the task so that it reads real frames.
        mock_task = mocker.Mock(spec=Task)
        mock_task.get_image.side_effect = [frame_1, frame_2]

        mock_job = mocker.Mock(spec=Job)
        mock_job.annotations_for_frame.return_value = annotations
        mock_task.jobs = [mock_job]

        # Create the tracker.
        _tracker = tracker.Tracker(mock_task)

        # Act.
        _tracker.track_forward()

        # Assert.
        # Serialize the new annotation points.
        args = [a for a, _ in mock_job.update_annotations.call_args_list]
        annotation_points = [a.points for a in args[0]]
        serial_points = yaml.dump(annotation_points, Dumper=yaml.Dumper)

        snapshot.assert_match(serial_points, "annotation_points.yaml")
