"""
Tests for the `job` module.
"""


import enum
import unittest.mock as mock

import pytest
from faker import Faker
from pydantic.dataclasses import dataclass
from pytest_mock import MockFixture
from swagger_client import ApiClient

from pycvat.dataset import job
from pycvat.type_helpers import ArbitraryTypesConfig


class TestJob:
    """
    Tests for the `Job` class.
    """

    @enum.unique
    class AnnotationType(enum.IntEnum):
        """
        Represents different types of annotations.
        """

        TAG = enum.auto()
        SHAPE = enum.auto()
        TRACK = enum.auto()

    @dataclass(frozen=True, config=ArbitraryTypesConfig)
    class ConfigForTests:
        """
        Encapsulates standard configuration for most tests.

        Attributes:
            job: The `Job` under test.
            mock_jobs_api_class: The mocked `JobsApi` class.
            mock_labeled_data_class: The mocked `LabeledData` class.
            mock_api_client: The mocked ApiClient instance.
            job_id: The job ID that we used when creating this class.
        """

        job: job.Job
        mock_jobs_api_class: mock.Mock
        mock_labeled_data_class: mock.Mock
        mock_api_client: ApiClient

        job_id: int

    @classmethod
    @pytest.fixture
    def config(cls, mocker: MockFixture, faker: Faker) -> ConfigForTests:
        """
        Generates standard configuration for most tests.

        Args:
            mocker: The fixture to use for mocking.
            faker: The fixture to use for generating fake data.

        Returns:
            The configuration that it created.

        """
        # Mock the dependencies.
        mock_jobs_api_class = mocker.patch(job.__name__ + ".JobsApi")
        mock_labeled_data_class = mocker.patch(job.__name__ + ".LabeledData")
        mock_api_client = mocker.create_autospec(ApiClient, instance=True)

        job_id = faker.random_int()

        job_ = job.Job(job_id=job_id, api_client=mock_api_client)

        return cls.ConfigForTests(
            job=job_,
            mock_jobs_api_class=mock_jobs_api_class,
            mock_labeled_data_class=mock_labeled_data_class,
            mock_api_client=mock_api_client,
            job_id=job_id,
        )

    def test_annotations_for_frame(
        self, config: ConfigForTests, faker: Faker
    ) -> None:
        """
        Tests that `annotations_for_frame` works.

        Args:
            config: The configuration to use for testing.
            faker: The fixture to use for generating fake data.

        """
        # Arrange.
        # Create some fake annotation data.
        mock_labeled_data = faker.labeled_data(min_frame=0, max_frame=0)
        # Make it look like the API produces that data.
        mock_api = config.mock_jobs_api_class.return_value
        mock_api.jobs_annotations_read.return_value = mock_labeled_data

        # Act.
        got_annotations = config.job.annotations_for_frame(0)

        # Assert.
        # It should have gotten the correct annotations.
        for annotation in (
            mock_labeled_data.tags
            + mock_labeled_data.shapes
            + mock_labeled_data.tracks
        ):
            assert annotation in got_annotations

    @pytest.mark.parametrize(
        "start_at", [None, 2], ids=["beginning", "frame_2"]
    )
    def test_iter_annotations(
        self, config: ConfigForTests, faker: Faker, start_at: int
    ) -> None:
        """
        Tests that `iter_annotations` works.

        Args:
            config: The configuration to use for testing.
            faker: The fixture to use for generating fake data.
            start_at: The frame to start iterating at.

        """
        # Arrange.
        # Create some fake annotation data.
        mock_labeled_data = faker.labeled_data(min_frame=0, max_frame=5)
        # Make it look like the API produces that data.
        mock_api = config.mock_jobs_api_class.return_value
        mock_api.jobs_annotations_read.return_value = mock_labeled_data

        # Make sure it starts at the first frame when the starting frame
        # isn't specified.
        mock_job = faker.job(start_frame=0)
        mock_api.jobs_read.return_value = mock_job

        # Act.
        got_annotations = list(config.job.iter_annotations(start_at=start_at))

        # Assert.
        if start_at is None:
            # Make the math easier.
            start_at = 0

        for annotation in (
            mock_labeled_data.tags
            + mock_labeled_data.shapes
            + mock_labeled_data.tracks
        ):
            if annotation.frame < start_at:
                # We didn't get annotations for this frame.
                continue

            # It should be associated with the correct frame.
            output_index = annotation.frame - start_at
            assert annotation in got_annotations[output_index]

    @pytest.mark.parametrize(
        "annotation_type", AnnotationType, ids=[t.name for t in AnnotationType]
    )
    def test_update_annotations(
        self,
        config: ConfigForTests,
        faker: Faker,
        annotation_type: AnnotationType,
    ) -> None:
        """
        Tests that `update_annotations` works.

        Args:
            config: The configuration to use for testing.
            faker: The fixture to use for generating fake data.
            annotation_type: The type of annotation to add for this test.

        """
        # Arrange.
        # Make it look like we have no annotations to start.
        mock_labeled_data = faker.labeled_data(
            max_num_tags=0, max_num_shapes=0, max_num_tracks=0
        )
        # Make it look like the API produces that data.
        mock_api = config.mock_jobs_api_class.return_value
        mock_api.jobs_annotations_read.return_value = mock_labeled_data

        # Create a fake annotation to add.
        if annotation_type == self.AnnotationType.TAG:
            annotation1 = faker.labeled_image()
            annotation2 = faker.labeled_image(frame=annotation1.frame)
        elif annotation_type == self.AnnotationType.SHAPE:
            annotation1 = faker.labeled_shape()
            annotation2 = faker.labeled_shape(frame=annotation1.frame)
        else:
            annotation1 = faker.labeled_track()
            annotation2 = faker.labeled_track(frame=annotation1.frame)

        # Act.
        config.job.update_annotations(annotation1)
        config.job.update_annotations(annotation2)

        # Assert.
        # It should have added both annotations.
        got_annotations = config.job.annotations_for_frame(annotation1.frame)
        assert annotation1 in got_annotations
        assert annotation2 in got_annotations

    def test_upload(self, config: ConfigForTests, faker: Faker) -> None:
        """
        Tests that `upload` works.

        Args:
            config: The configuration to use for testing.
            faker: The fixture to use for generating fake data.

        """
        # Arrange.
        # Create some fake annotation data.
        mock_labeled_data = faker.labeled_data()
        # Make it look like the API produces that data.
        mock_api = config.mock_jobs_api_class.return_value
        mock_api.jobs_annotations_read.return_value = mock_labeled_data

        # Act.
        config.job.upload()

        # Assert.
        # It should have created a new LabeledData object.
        config.mock_labeled_data_class.assert_called_once_with(
            version=mock_labeled_data.version,
            tags=mock_labeled_data.tags,
            shapes=mock_labeled_data.shapes,
            tracks=mock_labeled_data.tracks,
        )
        # It should have uploaded it.
        mock_api.jobs_annotations_update.assert_called_once_with(
            config.mock_labeled_data_class.return_value, config.job_id
        )

    def test_reload(self, config: ConfigForTests) -> None:
        """
        Tests that `reload` works.

        Args:
            config: The configuration to use for testing.

        """
        # Arrange.
        # Do something initially that will force data to be cached.
        (lambda: config.job.start_frame)()

        # Act.
        config.job.reload()

        # Assert.
        # Do it again and make sure it reloaded.
        (lambda: config.job.start_frame)()

        mock_api = config.mock_jobs_api_class.return_value
        assert mock_api.jobs_read.call_count == 2

    def test_start_frame(self, config: ConfigForTests, faker: Faker) -> None:
        """
        Tests that `start_frame` works.

        Args:
            config: The configuration to use for testing.
            faker: The fixture to use for generating fake data.

        """
        # Arrange.
        # Set the correct starting frame in the API response.
        start_frame = faker.random_int()
        mock_job = faker.job(start_frame=start_frame)
        mock_api = config.mock_jobs_api_class.return_value
        mock_api.jobs_read.return_value = mock_job

        # Act and assert.
        assert config.job.start_frame == start_frame

    def test_end_frame(self, config: ConfigForTests, faker: Faker) -> None:
        """
        Tests that `end_frame` works.

        Args:
            config: The configuration to use for testing.
            faker: The fixture to use for generating fake data.

        """
        # Arrange.
        # Set the correct starting frame in the API response.
        end_frame = faker.random_int()
        mock_job = faker.job(stop_frame=end_frame)
        mock_api = config.mock_jobs_api_class.return_value
        mock_api.jobs_read.return_value = mock_job

        # Act and assert.
        assert config.job.end_frame == end_frame
