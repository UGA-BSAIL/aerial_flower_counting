"""
Miscellaneous utilities for dealing with CVAT.
"""


from loguru import logger
from pycvat import Job, Task
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


@retry(
    retry=retry_if_exception_type(IOError),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(10),
    after=after_log(logger, "WARNING"),
)
def get_main_job(task: Task) -> Job:
    """
    We expect the tasks we're dealing with to have just one associated job. This
    gets that particular job.

    Args:
        task: The task.

    Returns:
        The job associated with that task.

    """
    jobs = task.get_jobs()
    if len(jobs) != 1:
        # Force it to reload job data if we retry.
        task.reload()
        # We have the wrong number of jobs. This might mean the CVAT server
        # simply hasn't registered a new job yet.
        raise IOError(f"Expected one job but have {len(jobs)}.")

    return jobs[0]
