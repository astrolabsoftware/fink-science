import numpy as np

def normalize_lc(lc_array: np.array) -> np.array:
    """
    Normalize a given array light curves.

    Parameters:
    ----------
    lc_array: np.array[float]
        Input light curve of an alert
        
    Returns:
    --------
    result: np.array[float]
        normalized light curve of an alert.
    """

    result = np.zeros((lc_array.shape[0], 3))
    result[:, 0] = lc_array[:, 0]
    result[:, 0] -= lc_array[0, 0]
    for i in range(1,3):
        norm = (lc_array[:,i] - np.min(lc_array[:,i])) / np.ptp(lc_array[:,i])
        result[:, i] = norm

    return result

