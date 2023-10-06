from kedro.io.partitioned_dataset import PartitionedDataset
from kedro.io import AbstractDataSet
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from typing import Any, Callable
from copy import deepcopy
from multiprocessing import cpu_count
from functools import partial


def _run_and_save(
    partition_data: Callable[[], Any], dataset: AbstractDataSet
) -> None:
    """
    Run the callable and save the result.

    Args:
        partition_data: Callable that returns the data to save.
        dataset: The dataset to save the data to.
    """
    data = partition_data()
    dataset.save(data)


class ParallelPartitionedDataSet(PartitionedDataset):
    """
    Extension to `PartitionedDataset` that can save items in parallel.
    """

    def __init__(
        self, *args: Any, max_workers: int = cpu_count(), **kwargs: Any
    ):
        """
        Args:
            *args: Forwarded to superclass.
            max_workers: The maximum number of workers. Defaults to the
                number of CPUs.
            **kwargs: Forwarded to superclass.
        """
        super().__init__(*args, **kwargs)

        self.__num_processes = max_workers
        self.__pool = ProcessPoolExecutor(max_workers=self.__num_processes)

    def _save(self, data: dict[str, Any]) -> None:
        if self._overwrite and self._filesystem.exists(self._normalized_path):
            self._filesystem.rm(self._normalized_path, recursive=True)

        running_jobs = set()
        for partition_id, partition_data in sorted(data.items()):
            kwargs = deepcopy(self._dataset_config)
            partition = self._partition_to_path(partition_id)
            # join the protocol back since tools like PySpark may rely on it
            kwargs[self._filepath_arg] = self._join_protocol(partition)
            dataset = self._dataset_type(**kwargs)  # type: ignore

            done_jobs = set()
            if len(running_jobs) >= self.__num_processes:
                # We need to wait for a job to finish before we can start a
                # new one.
                done_jobs, _ = wait(running_jobs, return_when=FIRST_COMPLETED)

            if not callable(partition_data):
                # If it's not a callable, we can't parallelize it, and should
                # just save it immediately.
                dataset.save(partition_data)
            else:
                new_job = self.__pool.submit(
                    partial(_run_and_save, partition_data, dataset)
                )
                running_jobs.add(new_job)

            # Find completed jobs to remove and save.
            running_jobs -= done_jobs

        # Make sure any remaining jobs are saved.
        wait(running_jobs)

        self._invalidate_caches()
