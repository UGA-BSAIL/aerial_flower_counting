"""
Utilities for fast spatial computations on shapes.
"""


from shapely import Polygon, prepare, minimum_bounding_radius
from typing import Iterable, List
from scipy.spatial import KDTree
import numpy as np


class ShapeOverlapChecker:
    """
    Handles checking overlaps between shapes.
    """

    def __init__(self, shapes: Iterable[Polygon]):
        """
        Args:
            shapes: The shapes to manage. We will be able to determine
                overlaps between any two shapes in this set.

        """
        self.__shapes = [prepare(s) for s in shapes]

        # Tree to use for fast rough overlap calculations.
        centroids = [s.centroid.coords for s in self.__shapes]
        self.__tree = KDTree(np.array(centroids).squeeze())

        # Determine the maximum of the minimum bounding radii for all the
        # points. This puts an upper bound on the distance we have to search
        # for overlap calculations.
        self.__max_radius = max(
            minimum_bounding_radius(s) for s in self.__shapes
        )

    def find_overlapping(self, shape: Polygon) -> List[Polygon]:
        """
        Finds all the shapes that overlap this one.

        Args:
            shape: The shape to check.

        Returns:
            The overlapping shapes.

        """
        # First, do a rough lookup using the tree. We can calculate an upper
        # bound for the distance we have to search.
        search_radius = self.__max_radius + minimum_bounding_radius(shape)
        # Find everything in that radius.
        nearby_indices = self.__tree.query_ball_point(
            shape.centroid.coords, search_radius, workers=-1
        )
        nearby_shapes = [self.__shapes[i] for i in nearby_indices]

        # Now we have to check for actual overlap.
        return [s for s in nearby_shapes if s.intersects(shape)]
