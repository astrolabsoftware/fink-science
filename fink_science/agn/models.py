import numpy as np

def protected_exponent(x1):
    """Exponential function : cannot exceed e**10
    """
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 10, np.exp(x1), np.exp(10))


def sig(x):
    """Sigmoid function using the protected exponential function
    """
    return 1/(1+protected_exponent(-x))


def bump(x, p1, p2, p3, p4):
    """ Parametric function, fit transient behavior
    
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
    """
    
    return sig(p1*x + p2 - protected_exponent(p3*x)) + p4


