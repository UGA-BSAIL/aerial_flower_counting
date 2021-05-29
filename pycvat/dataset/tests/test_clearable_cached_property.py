"""
Tests for the `clearable_cached_property` module.
"""


from pycvat.dataset import clearable_cached_property


class TestClearableCachedProperty:
    """
    Tests for the `ClearableCachedProperty` decorator.
    """

    def test_flush_cache(self) -> None:
        """
        Tests that we can correctly flush the cache.

        """
        # Arrange.
        # Create a class with a decorated property.
        class TestClass:
            def __init__(self):
                self.__counter = 0

            @clearable_cached_property.ClearableCachedProperty
            def property(self) -> int:
                self.__counter += 1
                return self.__counter

        test_class = TestClass()

        # Reading the property value should cache it.
        (lambda: test_class.property)()

        # Act.
        # Clear the cache.
        TestClass.property.flush_cache(test_class)

        # Assert.
        # Reading it again should get a new value.
        assert test_class.property == 2
        # However, it should once again be cached.
        assert test_class.property == 2
