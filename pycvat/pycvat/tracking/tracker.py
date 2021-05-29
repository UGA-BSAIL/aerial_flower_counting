"""
Tracks existing annotation points in subsequent frames using optical flow.
"""


import itertools
from typing import List, Tuple

import cv2
import numpy as np
from loguru import logger
from swagger_client import LabeledShape

from ..dataset.task import Task


class Tracker:
    """
    Tracks existing annotation points in subsequent frames using feature
    matching.
    """

    _AUTO_TRACK_ATTRIBUTE_NAME = "auto_tracked"
    """
    Name of the attribute to add to points that are automatically generated
    through tracking.
    """
    _COLORS = np.random.randint(0, 255, (100, 3))
    """
    Random colors to use when drawing.
    """
    _MATCH_THRESHOLD = 0.7
    """
    Scale threshold to use for determining whether a match is good or not.
    """
    _RANSAC_THRESHOLD = 2.0
    """
    Maximum RANSAC re-projection error.
    """
    _K_VALUE = 2
    """
    Value of K for KNN matching.
    """

    def __init__(self, task: Task, job_num: int = 0):
        """
        Args:
            task: The `Task` to source data from for tracking.
            job_num: The job number within this task to extract data from.
        """
        self.__task = task
        self.__job_num = job_num

        # Create a SIFT extractor and matcher to use.
        self.__sift = cv2.SIFT_create()
        self.__matcher = cv2.FlannBasedMatcher_create()

    # Not covered because this is purely for debugging.
    @classmethod
    def __plot_annotations(
        cls,
        *,
        old_points: np.ndarray,
        new_points: np.ndarray,
        previous_frame: np.ndarray,
        next_frame: np.ndarray,
    ) -> None:  # pragma: no cover
        """
        Debugging method that plots the original and projected annotations.

        Args:
            old_points: The original annotation points.
            new_points: The projected annotation points.
            previous_frame: The first frame.
            next_frame: The subsequent frame.

        """
        # Copy the frames so they are not modified.
        previous_frame = previous_frame.copy()
        next_frame = next_frame.copy()

        def draw_annotations(
            frame: np.ndarray, points: np.ndarray
        ) -> np.ndarray:
            for i, point in enumerate(points):
                color = cls._COLORS[i].tolist()
                point = tuple(point.astype(np.int32).squeeze())

                cv2.circle(frame, point, 10, color, 5)

            # Resize for easier display.
            return cv2.resize(frame, (1280, 720))

        previous_frame = draw_annotations(previous_frame, old_points)
        next_frame = draw_annotations(next_frame, new_points)

        # Display the frames.
        cv2.imshow("First Frame", previous_frame)
        cv2.imshow("Second Frame", next_frame)
        cv2.waitKey()

    def __match_features(
        self, previous_frame: np.ndarray, next_frame: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch]]:
        """
        Extracts features from two frames and matches them.

        Args:
            previous_frame: The first frame.
            next_frame: The subsequent frame.

        Returns:
            The list of features in the first frame, the list of features in
            the second frame, and the matches between them.

        """
        logger.debug("Extracting SIFT features.")
        previous_features, previous_descriptors = self.__sift.detectAndCompute(
            previous_frame, None
        )
        next_features, next_descriptors = self.__sift.detectAndCompute(
            next_frame, None
        )

        logger.debug("Finding feature matches.")
        matches = self.__matcher.knnMatch(
            previous_descriptors, next_descriptors, k=self._K_VALUE
        )

        # Filter all the good matches.
        matches = [
            m
            for m, n in matches
            if m.distance < self._MATCH_THRESHOLD * n.distance
        ]
        return previous_features, next_features, matches

    def __find_projection(
        self, previous_frame: np.ndarray, next_frame: np.ndarray
    ) -> np.ndarray:
        """
        Finds the projection matrix from one frame to the next.

        Args:
            previous_frame: The first frame.
            next_frame: The subsequent frame.

        Returns:
            The projection matrix from the first frame to the next one.

        """
        # Match features.
        source_features, dest_features, matches = self.__match_features(
            previous_frame, next_frame
        )

        # Find the most likely projection.
        logger.debug("Finding projection.")
        source_points = np.float32(
            [source_features[m.queryIdx].pt for m in matches]
        )
        dest_points = np.float32(
            [dest_features[m.trainIdx].pt for m in matches]
        )

        projection, _ = cv2.findHomography(
            source_points, dest_points, cv2.RANSAC, self._RANSAC_THRESHOLD
        )
        return projection

    def __track_one_step(
        self,
        previous_frame: np.ndarray,
        next_frame: np.ndarray,
        points: np.ndarray,
    ) -> np.ndarray:
        """
        Tracks a set of points forward one step.

        Args:
            previous_frame: The first frame that the point appears in.
            next_frame: The frame after that.
            points: The initial points.

        Returns:
            The location of the points in `next_frame`.
        """
        # Downsize the frames to speed up the matching process.
        previous_frame_small = cv2.pyrDown(previous_frame)
        next_frame_small = cv2.pyrDown(next_frame)

        # Calculate the projection matrix between the two frames.
        projection = self.__find_projection(
            previous_frame_small, next_frame_small
        )

        # Use it to project the annotations. Since our images were
        # down-sampled, we have to transform the points as well.
        points_small = points / 2
        return cv2.perspectiveTransform(points_small, projection) * 2

    def __track_annotations(
        self,
        previous_frame: np.ndarray,
        next_frame: np.ndarray,
        annotations: List[LabeledShape],
        *,
        next_frame_num: int,
        show_result: bool = False,
    ) -> List[LabeledShape]:
        """
        Performs a single tracking step.

        Args:
            previous_frame: The first frame that the annotations appear in.
            next_frame: The frame after that.
            annotations: The annotations that appear in the first frame.
            next_frame_num: The frame number to use for the new annotations.
            show_result: Whether to display the tracking result for debugging
                purposes.

        Returns:
            The propagated annotations that should appear in the next frame.

        """
        # Convert to OpenCV format.
        cv_points = self.__annotations_to_cv(annotations)

        # Perform the tracking.
        next_points = self.__track_one_step(
            previous_frame, next_frame, cv_points
        )

        # Not tested because this is only for debugging.
        if show_result:  # pragma: no cover
            # Display the tracking result.
            self.__plot_annotations(
                old_points=cv_points,
                new_points=next_points,
                previous_frame=previous_frame,
                next_frame=next_frame,
            )

        # Convert back to the CVAT format.
        return self.__cv_to_annotations(
            cv_points=next_points,
            original_annotations=annotations,
            frame_num=next_frame_num,
        )

    @staticmethod
    def __annotations_to_cv(annotations: List[LabeledShape]) -> np.ndarray:
        """
        Converts a list of annotations to a set of points that OpenCV can
        understand.

        Args:
            annotations: The annotations for several frames.

        Returns:
            The annotations in OpenCV form.

        """
        points = [a.points for a in annotations]
        # Flatten the points.
        points = list(itertools.chain.from_iterable(points))
        points_array = np.array(points, dtype=np.float32)
        # Restructure to a 3D array. (OpenCV expects an extra dimension.)
        return points_array.reshape((len(points_array) // 2, 1, 2))

    @classmethod
    def __cv_to_annotations(
        cls,
        *,
        cv_points: np.ndarray,
        frame_num: int,
        original_annotations: List[LabeledShape],
    ) -> List[LabeledShape]:
        """
        Converts a set of points from OpenCV to the CVAT annotations format.

        Args:
            cv_points: The points from OpenCV.
            frame_num: The frame number to set for the new annotations.
            original_annotations: The original annotations for these points.
                Should correspond exactly to the OpenCV variants.

        Returns:
            A list of CVAT annotations.

        """
        annotations = []
        flat_points = cv_points.flat
        flat_point_index = 0

        for old_annotation in original_annotations:
            # Make sure to grab the correct number of points for this
            # annotation.
            num_points = len(old_annotation.points)
            new_points = flat_points[
                flat_point_index : (flat_point_index + num_points)
            ]
            flat_point_index += num_points

            # Carry over all the metadata from the original annotation.
            annotation = LabeledShape(
                type=old_annotation.type,
                occluded=old_annotation.occluded,
                z_order=old_annotation.z_order,
                points=new_points.tolist(),
                frame=frame_num,
                label_id=old_annotation.label_id,
                group=old_annotation.group,
                source=old_annotation.source,
                attributes=old_annotation.attributes,
            )
            annotations.append(annotation)

        return annotations

    def track_forward(
        self, start_frame: int = 0, show_result: bool = False
    ) -> List[LabeledShape]:
        """
        Tracks each annotation in the initial frame forward by one frame. The
        updated annotations will automatically be saved in the cvat.

        Args:
            start_frame: The frame number to start tracking at.
            show_result: Whether to display the tracking result for debugging
                purposes.

        Returns:
            An amended list of annotations for the subsequent frame.

        """
        job = self.__task.jobs[self.__job_num]

        first_frame = self.__task.get_image(start_frame)
        first_annotations = job.annotations_for_frame(start_frame)
        next_frame = self.__task.get_image(start_frame + 1)

        # Propagate the annotations forward.
        propagated_annotations = self.__track_annotations(
            first_frame,
            next_frame,
            first_annotations,
            show_result=show_result,
            next_frame_num=start_frame + 1,
        )

        # Save the annotations.
        for annotation in propagated_annotations:
            job.update_annotations(annotation)

        return job.annotations_for_frame(start_frame + 1)
