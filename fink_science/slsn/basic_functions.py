import numpy as np
import pandas as pd  # noqa: F401


def compute_mean(x):
    """Compute mean of an array.
    Return -1 if the array is empty.

    Parameters
    ----------
    x: np.array

    Returns
    -------
    float
        Mean of the array.

    Examples
    --------
    >>> compute_mean(np.array([0, 1, 2, 3, 4])) == 2
    True
    >>> compute_mean(np.array([])) == -1
    True
    """

    if len(x) == 0:
        return -1
    else:
        return np.mean(x)


def compute_std(x):
    """Compute standard deviation of an array.
    Return -1 if the array is empty.

    Parameters
    ----------
    x: np.array

    Returns
    -------
    float
        Standard deviation of the array.

    Examples
    --------
    >>> compute_std(np.array([0, 1, 2, 3, 4])) == 1.4142135623730951
    True
    >>> compute_std(np.array([])) == -1
    True
    """

    if len(x) == 0:
        return -1
    else:
        return np.std(x)


def keep_filter(ps, band):
    """
    Funtion that removes points from other bands than the one specified

    Parameters
    ---------
    ps: pd.Series
        each rows of the dataframe. each entries must be numeric list.
    band: int
        Integer associated with the filter to keep.

    Return
    ------
    list
        list of the same size as x, each entries is the original list from the
        current rows with only the wanted filter and the associated values from the other columns.

    Example
    -------
    >>> example = pd.Series(data = {'cfilterName':np.array([1, 1, 2, 1, 2]), 'anything':np.array([-2, 86.9, 58.1, 24, 42])})
    >>> filtered = keep_filter(example, 2)
    >>> (np.array_equal(filtered[0], np.array([2, 2]))) & (np.array_equal(filtered[1], np.array([58.1, 42])))
    True
    >>> example2 = pd.Series(data = {'cfilterName':np.array([2, 2]), 'anything':np.array([24, 42])})
    >>> filtered2 = keep_filter(example2, 1)
    >>> (np.array_equal(filtered2[0], np.array([]))) & (np.array_equal(filtered2[1], np.array([])))
    True

    """

    mask = ps["cfilterName"] == band

    return [np.array(_col)[mask].astype(type(_col[0])) for _col in ps]


def get_max(x):
    """Returns maximum of an array. Returns -1 if array is empty

    Parameters
    ----------
    x: np.array

    Returns
    -------
    float
        Maximum of the array or -1 if array is empty.

    Example
    -------
    >>> get_max(np.array([1, 78, -6])) == 78
    True
    >>> get_max(np.array([])) == -1
    True
    """

    if len(x) == 0:
        return -1

    return np.max(x)


def translate(ps):
    """Translate a cmidPointTai list by substracting maxflux point

    Parameters
    ----------
    ps: pd.Series
        Must contain ['cmidPointTai', 'cpsFlux']

    Returns
    -------
    np.array
        Translated array. Returns empty array if input was empty

    Example
    -------
    >>> example = pd.Series(data = {'cmidPointTai':np.array([1,2,3]), 'cpsFlux':np.array([-2, 42, 23]), 'anything':np.array(['toto', 82, -8])})
    >>> np.array_equal(translate(example), np.array([-1,  0,  1]))
    True
    >>> example2 = pd.Series(data = {'cmidPointTai':np.array([]), 'cpsFlux':np.array([]), 'anything':np.array(['toto', 82, -8])})
    >>> np.array_equal(translate(example2), np.array([]))
    True

    """

    if len(ps["cmidPointTai"]) == 0:
        return []

    else:
        return (
            np.array(ps["cmidPointTai"]) - ps["cmidPointTai"][np.argmax(ps["cpsFlux"])]
        )


def normalize(ps):
    """Normalize by dividing by a data frame of maximum

    Parameters
    ----------
    ps: pd.Series
        Must contain 'cpsFlux', 'cpsFluxErr' and 'peak'

    Returns
    -------
    pd.Series
        Dataframe with columns 'cpsFlux' and 'cpsFluxErr' normalized

    Example
    -------
    >>> example = pd.Series(data = {'cpsFlux':np.array([17, 35.7, -3]), 'cpsFluxErr':np.array([0.7, 1, 0]), 'peak':35.7})
    >>> out = normalize(example)
    >>> np.array_equal(np.round(out[0], 3), np.array([ 0.476,  1.   , -0.084]))
    True
    >>> np.array_equal(np.round(out[1], 3), np.array([0.02 , 0.028, 0.   ]))
    True
    >>> example2 = pd.Series(data = {'cpsFlux':np.array([]), 'cpsFluxErr':np.array([]), 'peak':-1})
    >>> out2 = normalize(example2)
    >>> (np.array_equal(out2[0], np.array([]))) & (np.array_equal(out2[1], np.array([])))
    True
    """

    if len(ps["cpsFlux"]) == 0:
        return ps[["cpsFlux", "cpsFluxErr"]]

    else:
        ps["cpsFlux"] = np.array(ps["cpsFlux"]) / np.array(ps["peak"])
        ps["cpsFluxErr"] = np.array(ps["cpsFluxErr"]) / np.array(ps["peak"])
        return ps[["cpsFlux", "cpsFluxErr"]]
