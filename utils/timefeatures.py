"""
Time feature extraction for temporal embeddings.

Converts timestamps to normalized numerical features used as temporal marks
in dataset classes (seq_x_mark, seq_y_mark).
"""

import numpy as np
import pandas as pd


def time_features(dates: pd.Series) -> np.ndarray:
    """Convert a Series of timestamps to normalized time features.

    Each timestamp is decomposed into 4 cyclic/linear features, all
    normalized to the range [0, 1]:

        col 0: hour of day    / 23.0   (0 = midnight, 1 = 23:00)
        col 1: day of month   / 30.0   (0 = 1st, 1 = 31st mapped to 30)
        col 2: day of week    / 6.0    (0 = Monday, 1 = Sunday)
        col 3: month of year  / 11.0   (0 = January, 1 = December)

    Args:
        dates: pd.Series of timestamps (datetime-like or string-parseable).

    Returns:
        np.ndarray of shape (N, 4) and dtype float32.
    """
    # Always convert via pd.to_datetime — handles datetime64, strings, and
    # tz-aware types uniformly without fragile dtype introspection.
    dates = pd.to_datetime(dates)

    dt = pd.DatetimeIndex(dates)

    features = np.column_stack(
        [
            dt.hour / 23.0,
            (dt.day - 1) / 30.0,
            dt.dayofweek / 6.0,
            (dt.month - 1) / 11.0,
        ]
    ).astype(np.float32)

    assert features.shape == (len(dates), 4), (
        f"Expected shape ({len(dates)}, 4), got {features.shape}"
    )
    return features
