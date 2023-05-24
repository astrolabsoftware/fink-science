import numpy as np

def extract_max_prob(arr: list) -> list:
    """ Extract main class and associated probability from a vector of probabilities
    """
    if np.isnan(arr[0]):
        return [-1, 0.0]
    array = np.array(arr)
    index = np.argmax(array)
    return [index, array[index]]

def norm_column(col: list) -> np.array:

    col = np.array(col)

    if len(col) == 1:
        norm = [1.0]
    else:
        norm = (col - col.min()) / np.ptp(col)

    return norm
