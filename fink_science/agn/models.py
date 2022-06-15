import numpy as np


def protected_exponent(x1):
    """Exponential function : cannot exceed e**10
    
    Examples
    --------
    >>> np.round_(protected_exponent(42), 5)
    22026.46579
    >>> np.round_(protected_exponent(1), 5)
    2.71828
    """
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 10, np.exp(x1), np.exp(10))


def sig(x):
    """Sigmoid function using the protected exponential function
    
    Examples
    --------
    >>> sig(0)
    0.5
    """
    
    return 1 / (1 + protected_exponent(-x))


def bump(x, p1, p2, p3, p4):
    """Parametric function, fit transient behavior

    Parameters
    ----------
    x : np.array
        Array of mjd translated to 0
    p1,p2,p3, p4 : floats
        Parameters of the function

    Returns
    -------
    np.array
        Flux array
        
    Examples
    --------    
    >>> np.round_(bump(0, 0.225,-2.5,0.038,0), 5)
    0.02931
    """

    return sig(p1 * x + p2 - protected_exponent(p3 * x)) + p4

if __name__ == "__main__":
    
    import sys
    import doctest
    
    sys.exit(doctest.testmod()[0])