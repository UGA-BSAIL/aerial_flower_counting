from typing import Any

from backports.cached_property import cached_property


class ClearableCachedProperty(cached_property):
    """
    An extension of a cached property that can be manually cleared.
    """

    def flush_cache(self, instance: Any) -> None:
        """
        Clears the cached value of this property, if it is present, forcing
        it to be recalculated.

        Args:
            instance: The instance to clear the cache for.

        """
        with self.lock:
            instance.__dict__.pop(self.attrname, None)
