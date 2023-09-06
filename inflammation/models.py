"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np
from functools import reduce

def load_csv(filename):
    """Load a Numpy array from a CSV
    :param filename: Filename of CSV to load

    Check this array is 2D, >=0, no NaNs
    """
    tdata=np.loadtxt(fname=filename, delimiter=',')
    if not isinstance(tdata, np.ndarray):
        raise TypeError('data input should be ndarray')
    if len(tdata.shape)!=2:
        raise ValueError('inflammation array should be 2D')
    if np.any(tdata < 0):
        raise ValueError('inflammation values should be >=0')
    if np.any(tdata > 20):
        raise ValueError('inflammation values should be <=20')
    if np.any(np.isnan(tdata)):
        raise ValueError('inflammation values should not contain NaN')

    return tdata


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array.
    :param data: A 2D data array with inflammation data \
    (each row contains measurements for a single patient\
    across all days).
    :returns: An array of mean values of measurements\
    for each day.
    """
    if np.any(data <0):
        raise ValueError('Sane inflammation values should be >=0')

    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array.
    :param data: A 2D data array with inflammation data \
    (each row contains measurements for a single patient\
    across all days).
    :returns: An array of max values of measurements\
    for each day.
    """
    if np.any(data <0):
        raise ValueError('Sane inflammation values should be >=0')

    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array.
    :param data: A 2D data array with inflammation data \
    (each row contains measurements for a single patient\
    across all days).
    :returns: An array of min values of measurements\
    for each day.
    """
    if np.any(data <0):
        raise ValueError('Sane inflammation values should be >=0')


    return np.min(data, axis=0)


def daily_std(data):
    """Calculate the daily stdev of a 2D inflammation data array.
    :param data: A 2D data array with inflammation data \
    (each row contains measurements for a single patient\
    across all days).
    :returns: An array of min values of measurements\
    for each day.
    """
    if np.any(data <0):
        raise ValueError('Sane inflammation values should be >=0')

    return np.std(data, axis=0)




def patient_normalise(data):
    """
    Normalise patient data from a 2D inflammation data array.

    NaN values are ignored, and normalised to 0.

    Negative values are rounded to 0.
    """

    if np.any(data <0):
        raise ValueError('Sane inflammation values should be >=0')

    max_data = np.nanmax(data, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised = data / max_data[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    normalised[normalised < 0] = 0
    return normalised

def daily_above_threshold(data, patient, threshold):
    """
        Return boolean list, true if
        daily inflammation data of patient exceeds threshold
    """
    pdata=data[patient]

    blist = list(map(lambda x: x > threshold, pdata))
    bcount= reduce(lambda a,b:a+1 if b else a, blist,0)
    return bcount