"""
Tests for the `task` module.
"""


import unittest.mock as mock
from pathlib import Path
from typing import List

import pytest
from faker import Faker
from pydantic.dataclasses import dataclass
from pytest_mock import MockFixture
from swagger_client import ApiClient, Label, Segment, SimpleJob
from swagger_client import Task as TaskModel

from pycvat.dataset import task
from pycvat.type_helpers import ArbitraryTypesConfig


class TestTask:
    """
    Tests for the `Task` class.
    """

    @dataclass(frozen=True, config=ArbitraryTypesConfig)
    class ConfigForTests:
        """
        Encapsulates standard configuration for most tests.

        Attributes:
            task: The `Task` under test.
            mock_job_class: The mocked `Job` class.
            mock_api_client: The mocked `ApiClient` to use for testing.
            mock_client_file_class: The mocked `ClientFile` class.
            mock_data_class: The mocked `Data` class.
            mock_tasks_api_class: The mocked `TaskApi` class.
            mock_task_model_class: The mocked `TaskModel` class.

            task_id: The fake task ID that we used to create the task.
        """

        task: task.Task
        mock_job_class: mock.Mock
        mock_api_client: ApiClient
        mock_client_file_class: mock.Mock
        mock_data_class: mock.Mock
        mock_tasks_api_class: mock.Mock
        mock_task_model_class: mock.Mock

        task_id: int

    @dataclass(frozen=True, config=ArbitraryTypesConfig)
    class JobMocks:
        """
        Encapsulates mocks for simulated job data.

        Attributes:
            mock_jobs: A list of mocked jobs.
            mock_task_model: The corresponding mocked `TaskModel` that
                contains those jobs.
        """

        mock_jobs: List[SimpleJob]
        mock_task_model: TaskModel

    @classmethod
    @pytest.fixture
    def config(cls, mocker: MockFixture, faker: Faker) -> ConfigForTests:
        """
        Generates standard configuration for most tests.

        Args:
            mocker: The fixture to use for mocking.
            faker: The fixture to use for generating fake data.

        Returns:
            The configuration that it generated.

        """
        # Mock the dependencies.
        mock_job_class = mocker.patch(task.__name__ + ".Job")
        mock_api_client = mocker.create_autospec(ApiClient, instance=True)
        mock_client_file_class = mocker.patch(task.__name__ + ".ClientFile")
        mock_data_class = mocker.patch(task.__name__ + ".Data")
        mock_tasks_api_class = mocker.patch(task.__name__ + ".TasksApi")
        mock_task_model_class = mocker.patch(task.__name__ + ".TaskModel")

        task_id = faker.random_int()

        _task = task.Task(task_id=task_id, api_client=mock_api_client)

        return cls.ConfigForTests(
            task=_task,
            mock_job_class=mock_job_class,
            mock_api_client=mock_api_client,
            mock_client_file_class=mock_client_file_class,
            mock_data_class=mock_data_class,
            mock_tasks_api_class=mock_tasks_api_class,
            mock_task_model_class=mock_task_model_class,
            task_id=task_id,
        )

    @classmethod
    @pytest.fixture
    def job_mocks(cls, mocker: MockFixture, faker: Faker) -> JobMocks:
        """
        Creates mocks for simulating jobs.

        Args:
            mocker: The fixture to use for mocking.
            faker: The fixture to use for generating fake data.

        Returns:
            The mocks that it created.

        """
        mock_job1 = faker.simple_job(job_id=0)
        mock_job2 = faker.simple_job(job_id=1)

        mock_segment = mocker.create_autospec(Segment, instance=True)
        mock_segment.jobs = [mock_job1, mock_job2]

        mock_task_model = mocker.create_autospec(TaskModel, instance=True)
        mock_task_model.segments = [mock_segment]

        return cls.JobMocks(
            mock_jobs=[mock_job1, mock_job2], mock_task_model=mock_task_model
        )

    @pytest.mark.parametrize(
        "compressed", [False, True], ids=["not_compressed", "compressed"]
    )
    def test_get_image(
        self,
        config: ConfigForTests,
        mocker: MockFixture,
        faker: Faker,
        compressed: bool,
    ) -> None:
        """
        Tests that `get_image` works.

        Args:
            config: The configuration to use for testing.
            mocker: The fixture to use for mocking.
            faker: The fixture to use for generating fake data.
            compressed: Whether to load the compressed image or the extracted
                one.

        """
        # Arrange.
        # Mock the opencv imdecode function.
        mock_imdecode = mocker.patch("cv2.imdecode")

        # Make it look like we read a byte string from CVAT.
        mock_tasks_api = config.mock_tasks_api_class.return_value
        fake_image = faker.binary(length=64)
        mock_task_model = mocker.create_autospec(TaskModel, instance=True)
        mock_task_model.data = fake_image
        mock_tasks_api.tasks_data_read.return_value = mock_task_model

        frame_num = faker.random_int()

        # Act.
        got_image = config.task.get_image(frame_num, compressed=compressed)

        # Assert.
        # It should have retrieved the frame from CVAT.
        mock_tasks_api.tasks_data_read.assert_called_once_with(
            config.task_id,
            mocker.ANY,
            mocker.ANY,
            frame_num,
            _preload_content=False,
        )

        if not compressed:
            # It should have decoded the image.
            mock_imdecode.assert_called_once()
            args, _ = mock_imdecode.call_args
            image = args[0]
            assert image.tobytes() == fake_image

            assert got_image == mock_imdecode.return_value
        else:
            # It should have read the raw image data.
            assert got_image.tobytes() == fake_image

    def test_get_image_size(
        self, config: ConfigForTests, faker: Faker,
    ) -> None:
        """
        Tests that `get_image_size` works.

        Args:
            config: The configuration to use for testing.
            faker: The fixture to use for generating fake data.

        """
        # Arrange.
        # Make it look like we have valid metadata.
        mock_tasks_api = config.mock_tasks_api_class.return_value
        mock_metadata = mock_tasks_api.tasks_data_data_info.return_value
        mock_frame_meta = faker.frame_meta()
        mock_metadata.frames = [mock_frame_meta]

        # Act.
        width, height = config.task.get_image_size(0)

        # Assert.
        assert width == mock_frame_meta.width
        assert height == mock_frame_meta.height

    def test_get_jobs(
        self, config: ConfigForTests, job_mocks: JobMocks
    ) -> None:
        """
        Tests that `get_jobs` works.

        Args:
            config: The configuration to use for testing.
            job_mocks: Mocks for simulating jobs.

        """
        # Arrange.
        # Make it look like we have some jobs.
        mock_tasks_api = config.mock_tasks_api_class.return_value
        mock_tasks_api.tasks_read.return_value = job_mocks.mock_task_model

        # Act.
        got_jobs = config.task.jobs

        # Assert.
        # It should have initialized the Job instances.
        for job_model in job_mocks.mock_jobs:
            config.mock_job_class.assert_any_call(
                job_id=job_model.id, api_client=config.mock_api_client
            )

        assert got_jobs == [config.mock_job_class.return_value] * len(
            job_mocks.mock_jobs
        )

    def test_get_labels(self, config: ConfigForTests, faker: Faker) -> None:
        """
        Tests that `get_labels` works.

        Args:
            config: The configuration to use for testing.
            faker: The fixture to use for creating fake data.

        """
        # Arrange.
        mock_api = config.mock_tasks_api_class.return_value
        mock_task_data = mock_api.tasks_read.return_value

        # Make it look like we have labels.
        mock_labels = [faker.label(), faker.label()]
        mock_task_data.labels = mock_labels

        # Act.
        got_labels = config.task.get_labels()

        # Assert.
        assert got_labels == mock_labels

    def test_num_frames(
        self, config: ConfigForTests, mocker: MockFixture
    ) -> None:
        """
        Tests that `num_frames` works.

        Args:
            config: The configuration to use for testing.
            mocker: The fixture to use for mocking.

        """
        # Arrange.
        mock_api = config.mock_tasks_api_class.return_value
        mock_task_data = mock_api.tasks_read.return_value

        # Make it look like we have some segments.
        mock_segment_1 = mocker.create_autospec(Segment, instance=True)
        mock_segment_2 = mocker.create_autospec(Segment, instance=True)
        mock_segment_1.start_frame = 1
        mock_segment_1.stop_frame = 1000
        mock_segment_2.start_frame = 800
        mock_segment_2.stop_frame = 1800
        mock_task_data.segments = [mock_segment_1, mock_segment_2]

        # Act and assert.
        assert config.task.num_frames == 1800

    def test_find_label(self, config: ConfigForTests, faker: Faker) -> None:
        """
        Tests that `find_label` works.

        Args:
            config: The configuration to use for testing.
            faker: The fixture to use for creating fake data.

        """
        # Arrange.
        mock_api = config.mock_tasks_api_class.return_value
        mock_task_data = mock_api.tasks_read.return_value

        # Make it look like we have labels.
        mock_labels = [
            faker.label(name="Jerry"),
            faker.label(name="Steve"),
        ]
        mock_task_data.labels = mock_labels

        # Act.
        jerry = config.task.find_label("Jerry")
        steve = config.task.find_label("Steve")

        # Assert.
        assert jerry.name == "Jerry"
        assert steve.name == "Steve"

    def test_find_label_nonexistent(self, config: ConfigForTests) -> None:
        """
        Tests that `find_label` fails when we try to find a label that
        doesn't exist.

        Args:
            config: The configuration to use for testing.

        """
        # Arrange.
        # Make it look like there are no labels.
        mock_api = config.mock_tasks_api_class.return_value
        mock_task_data = mock_api.tasks_read.return_value
        mock_task_data.labels = []

        # Act and assert.
        with pytest.raises(NameError, match="no label"):
            config.task.find_label("invalid")

    def test_find_image_frame_num(
        self, config: ConfigForTests, faker: Faker,
    ) -> None:
        """
        Tests that `find_image_frame_num` works.

        Args:
            config: The configuration to use for testing.
            faker: The fixture to use for creating fake data.

        """
        # Arrange.
        # Make it look like we have valid metadata.
        mock_tasks_api = config.mock_tasks_api_class.return_value
        mock_metadata = mock_tasks_api.tasks_data_data_info.return_value
        mock_frame_meta = faker.frame_meta()
        mock_metadata.frames = [faker.frame_meta(), mock_frame_meta]

        # Act.
        got_frame_num = config.task.find_image_frame_num(mock_frame_meta.name)

        # Assert.
        # It should have found the second frame.
        assert got_frame_num == 1

    def test_upload(self, config: ConfigForTests, job_mocks: JobMocks) -> None:
        """
        Tests that `upload` works.

        Args:
            config: The configuration to use for testing.
            job_mocks: Mocks for simulating jobs.

        """
        # Arrange.
        # Make it look like we have some jobs.
        mock_tasks_api = config.mock_tasks_api_class.return_value
        mock_tasks_api.tasks_read.return_value = job_mocks.mock_task_model

        # Act.
        config.task.upload()

        # Assert.
        # It should have uploaded each job.
        mock_job = config.mock_job_class.return_value
        assert mock_job.upload.call_count == 2

    def test_reload(self, config: ConfigForTests) -> None:
        """
        Tests that `reload` works.

        Args:
            config: The configuration to use for testing.

        """
        # Arrange.
        # Do something initially that will force data to be cached.
        config.task.get_labels()

        # Act.
        config.task.reload()

        # Assert.
        # Do it again and make sure it reloaded.
        config.task.get_labels()

        mock_api = config.mock_tasks_api_class.return_value
        assert mock_api.tasks_read.call_count == 2

    def test_id(self, config: ConfigForTests) -> None:
        """
        Tests that the `id` property works.

        Args:
            config: The configuration to use for testing.

        """
        # Act and assert.
        assert config.task.id == config.task_id

    def test_create_new(
        self, config: ConfigForTests, mocker: MockFixture, faker: Faker
    ) -> None:
        """
        Tests that `create_new` works.

        Args:
            config: The configuration to use for testing.
            mocker: The fixture to use for mocking.
            faker: The fixture to use for generating fake data.

        """
        # Arrange.
        # Mock the dependencies.
        mock_extended_api_class = mocker.patch(
            task.__name__ + ".ExtendedTasksApi"
        )

        # Mock some labels.
        mock_label1 = mocker.create_autospec(Label, instance=True)
        mock_label2 = mocker.create_autospec(Label, instance=True)

        # Create fake parameters.
        fake_name = faker.text(max_nb_chars=100)
        fake_bug_tracker = faker.url()
        fake_quality = faker.random_int(min=0, max=100)

        # Create fake images.
        images = []
        for _ in range(10):
            images.append(Path(faker.file_path(category="image")))

        # Act.
        task.Task.create_new(
            api_client=config.mock_api_client,
            name=fake_name,
            labels=[mock_label1, mock_label2],
            bug_tracker=fake_bug_tracker,
            images=images,
            image_quality=fake_quality,
        )

        # Assert.
        # It should have created the task on the server.
        config.mock_task_model_class.assert_called_once_with(
            name=fake_name,
            labels=[mock_label1, mock_label2],
            bug_tracker=fake_bug_tracker,
        )
        mock_task_model = config.mock_task_model_class.return_value

        mock_extended_api_class.assert_called_once_with(config.mock_api_client)
        mock_api = mock_extended_api_class.return_value
        mock_api.tasks_create.assert_called_once_with(mock_task_model)
        mock_created_task = mock_api.tasks_create.return_value

        # It should have added the images.
        for image in images:
            config.mock_client_file_class.assert_any_call(file=image)
        mock_client_file = config.mock_client_file_class.return_value

        config.mock_data_class.assert_called_once_with(
            image_quality=fake_quality,
            client_files=[mock_client_file] * len(images),
        )
        mock_data = config.mock_data_class.return_value

        mock_api.tasks_data_create.assert_called_once_with(
            mock_data, mock_created_task.id
        )
