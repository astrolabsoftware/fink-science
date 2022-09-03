import numpy as np
import pandas as pd

def normalize_lc(lc_array: np.array) -> np.array:
    """
    Normalize an array light curves.

    Parameters:
    ----------
    lc_array: np.array

    Returns
    -------
    out: np.array
        normalized light curve
    """

    result = np.zeros((lc_array.shape[0], 3))
    result[:, 0] = lc_array[:, 0]
    result[:, 0] -= lc_array[0, 0]
    for i in range(1,3):
        norm = (lc_array[:,i] - np.min(lc_array[:,i])) / np.ptp(lc_array[:,i])
        result[:, i] = norm

    return result

