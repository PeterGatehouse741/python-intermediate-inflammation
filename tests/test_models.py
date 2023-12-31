"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

@pytest.mark.parametrize(
    "test_mean, expected_mean, expect_raises",
    [
        ([ [0, 0], [0, 0], [0, 0] ],
         [0, 0],
         None),
        ([ [1, 2], [3, 4], [5, 6] ],
         [3, 4],
         None),
        ([[-1, 2], [3, 4], [5, 6]],
         [3, 4],
         ValueError)

    ])
def test_daily_mean(test_mean, expected_mean, expect_raises):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_mean
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            npt.assert_array_equal(daily_mean(np.array(test_mean)), np.array(expected_mean))
    else:
        npt.assert_array_equal(daily_mean(np.array(test_mean)), np.array(expected_mean))

@pytest.mark.parametrize(
    "test_min, expected_min, expect_raises",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0], None),
        ([ [1, 2], [3, 4], [5, 6] ], [1, 2], None),
        ([ [-1, 2], [3, 4], [5, 6] ], [1, 2], ValueError)
    ])
def test_daily_min(test_min, expected_min, expect_raises):
    """Test min function works for array of zeroes and positive integers."""
    from inflammation.models import daily_min
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            npt.assert_array_equal(daily_min(np.array(test_min)), np.array(expected_min))
    else:
        npt.assert_array_equal(daily_min(np.array(test_min)), np.array(expected_min))

@pytest.mark.parametrize(
    "test_max, expected_max, expect_raises",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0], None),
        ([ [1, 2], [3, 4], [5, 6] ], [5, 6], None),
        ([ [2, 1], [4, 3], [6, 5] ], [6, 5], None),
        ([ [-2, 1], [4, 3], [6, 5] ], [6, 5], ValueError)
    ])
def test_daily_max(test_max, expected_max, expect_raises):
    """Test min function works for array of zeroes and positive integers."""
    from inflammation.models import daily_max
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            npt.assert_array_equal(daily_max(np.array(test_max)), np.array(expected_max))
    else:
        npt.assert_array_equal(daily_max(np.array(test_max)), np.array(expected_max))


@pytest.mark.parametrize(
    "test_std, expected_std, expect_raises",
    [
        ([ [0, 0], [0, 0], [0, 0] ],
         [0, 0],
         None),
        ([ [1, 2], [3, 5], [5, 7] ],
         [1.633, 2.055],
         None),
        ([[-1, 2], [3, 4], [5, 6]],
         [3, 4],
         ValueError)

    ])
def test_daily_std(test_std, expected_std, expect_raises):
    """Parametrized test of daily_std function."""
    from inflammation.models import daily_std
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            npt.assert_array_almost_equal(daily_std(np.array(test_std)), np.array(expected_std),decimal=2)
    else:
        npt.assert_array_almost_equal(daily_std(np.array(test_std)), np.array(expected_std),decimal=2)



@pytest.mark.parametrize(
    "test_norm, expected_norm, expect_raises",
    [
        ( [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
          [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
          None),
        ( [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
          [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
          None),
        ( [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
          [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
          None),
        ([[1, 2, 3], [4, 5, float('nan')], [7, 8, 9]],
         [[0.33, 0.67, 1], [0.8, 1, 0], [0.78, 0.89, 1]],
         None),
        ([[1, 2, 3], [float('nan'),float('nan'), float('nan')], [7, 8, 9]],
         [[0.33, 0.67, 1], [0, 0, 0], [0.78, 0.89, 1]],
         None),
        ( [[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
          [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
          ValueError)

    ])

def test_patient_normalise(test_norm, expected_norm, expect_raises):
    """Test normalisation works for arrays of one and positive integers.
       Assumption that test accuracy of two decimal places is sufficient."""
    from inflammation.models import patient_normalise
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            npt.assert_array_almost_equal(patient_normalise(np.array(test_norm)), np.array(expected_norm),decimal=2)
    else:
        npt.assert_array_almost_equal(patient_normalise(np.array(test_norm)), np.array(expected_norm),decimal=2)


@pytest.mark.parametrize(
    "data_test, patient_test, threshold_test, expected_count",
    [
        ( [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
          1,
          5,
          0),
        ([[0, 0, 0], [5, 6, 7], [0, 0, 0]],
         1,
         5,
         2),
        ([[4, 5, 4], [0, 0, 0], [0, 0, 0]],
         0,
         4,
         1)
    ])


def test_daily_above_threshold(data_test, patient_test, threshold_test, expected_count):
    from inflammation.models import daily_above_threshold
    npt.assert_array_equal(daily_above_threshold(np.array(data_test),patient_test,threshold_test), np.array(expected_count))
