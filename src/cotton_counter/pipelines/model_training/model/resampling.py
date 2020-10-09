"""
Utilities for resampling datasets.
"""


from typing import Any, Callable, Tuple

import tensorflow as tf


def _filter_by_class(
    item_class: tf.Tensor, *, sample_probabilities: tf.Tensor
) -> tf.Tensor:
    """
    Filters items that fall into particular classes according to a certain
    distribution.

    Args:
        item_class: The class of the item being filtered. Should be a 0D int
            tensor.
        sample_probabilities: The per-class keep probabilities for filtering.
            Should be a 1D float tensor, where each item is the probability
            of accepting a sample of the corresponding class.

    Returns:
        A boolean 0D tensor, indicating whether to keep this item or not.

    """
    item_class = tf.ensure_shape(item_class, ())

    # Determine the keep probability for this class.
    keep_prob = sample_probabilities[item_class]

    # Randomly choose to keep it or not.
    token = tf.random.uniform(())
    return tf.cond(
        tf.less_equal(token, keep_prob),
        true_fn=lambda: tf.constant(True),
        false_fn=lambda: tf.constant(False),
    )


def _filter_for_even_distribution(
    item_class: tf.Tensor, *, initial_dist: tf.Tensor
) -> tf.Tensor:
    """
    Filters items that fall into particular classes in order to achieve a
    uniform distribution of classes.

    Args:
        item_class: The class of the item being filtered. Should be a 0D int
            tensor.
        initial_dist: The approximate class distribution of the input.

    Returns:
        A boolean 0D tensor, indicating whether to keep this item or not.

    """
    # Calculate the probability of keeping an item of each class.
    keep_probabilities = tf.reduce_min(initial_dist) / initial_dist

    return _filter_by_class(
        item_class, sample_probabilities=keep_probabilities
    )


StateType = Tuple[tf.Tensor, tf.Tensor]
"""
Alias for the type of the state for the distribution estimator. The first
element is a 1D float vector containing the current distribution estimate,
and the second item is the total number of elements we have seen.
"""


def _update_distribution_estimate(
    old_state: StateType, item: Any, *, classify: Callable[[Any], tf.Tensor],
) -> Tuple[StateType, Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    """
    Updates the running estimates of the class distribution for a dataset based
    on the next element in the dataset.

    Args:
        old_state: The current estimator state.
        item: The item from the dataset.
        classify: A function that takes a dataset element and returns an
            integer class for it in the range [0, number of classes).

    Returns:
        The new estimator state, and a tuple of the item class, the original
        item, and the running distribution estimate.

    """
    current_estimates, total_count = old_state
    item_class = classify(item)

    # Update the state based on the class.
    new_total_count = total_count + tf.constant(1, dtype=tf.int64)
    inverse_count = tf.constant(1.0) / tf.cast(new_total_count, tf.float32)

    def update_class_estimate(class_index: tf.Tensor) -> tf.Tensor:
        """
        Updates the estimate for a single class.

        Args:
            class_index: The index of that class.

        Returns:
            The updated estimate.

        """
        tf.ensure_shape(class_index, ())

        class_estimate = current_estimates[class_index]
        partial_estimate = class_estimate * (1 - inverse_count)
        return tf.cond(
            tf.equal(class_index, item_class),
            # We saw a new instance of this class.
            true_fn=lambda: partial_estimate + inverse_count,
            # We didn't see a new instance of this class.
            false_fn=lambda: partial_estimate,
        )

    # Generate new estimates for all classes.
    num_classes = tf.size(current_estimates)
    class_indices = tf.range(num_classes)
    new_estimates = tf.map_fn(
        update_class_estimate, class_indices, dtype=tf.float32
    )

    return (new_estimates, new_total_count), (item_class, item, new_estimates)


def _estimate_distribution(
    dataset: tf.data.Dataset,
    *,
    classify: Callable[[Any], tf.Tensor],
    num_classes: int,
) -> tf.data.Dataset:
    """
    Creates a dataset that contains estimates of the distribution of an input
    dataset.

    Args:
        dataset: The raw dataset to estimate the distribution of.
        classify: A function that takes a dataset element and returns an
            integer class for it in the range [0, number of classes).
        num_classes: The total number of classes that we have.

    Returns:
        A dataset that is the same length as `dataset`. Each element contains a
        tuple of the class, the original dataset element, and the estimated
        class distribution created by using up to that many
        elements of `dataset`.

    """
    initial_distribution = tf.zeros((num_classes,), dtype=tf.float32)
    initial_count = tf.zeros((), dtype=tf.int64)

    scanner = tf.data.experimental.scan(
        (initial_distribution, initial_count),
        lambda s, e: _update_distribution_estimate(s, e, classify=classify),
    )
    return dataset.apply(scanner)


def balance_distribution(
    dataset: tf.data.Dataset,
    *,
    classify: Callable[[Any], tf.Tensor],
    num_classes: int,
) -> tf.data.Dataset:
    """
    A dataset transformation that balances the distribution of various
    classes in a dataset. It does this through rejection resampling.

    Args:
        dataset: The dataset to balance.
        classify: A function that takes a dataset element and returns an
            integer class for it in the range [0, number of classes).
        num_classes: The total number of classes that we have.

    Returns:
        A transformed dataset that is properly balanced.

    """
    # Determine the filtering probabilities to use.
    data_with_dist = _estimate_distribution(
        dataset, classify=classify, num_classes=num_classes
    )

    # Filter based on the class.
    filtered = data_with_dist.filter(
        lambda c, e, d: _filter_for_even_distribution(c, initial_dist=d)
    )

    # Strip out the classes and distribution estimates.
    return filtered.map(lambda _, e, __: e)
