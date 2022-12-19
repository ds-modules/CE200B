import numpy as np

def np_arange(a:float, b:float, step:float):
    """
    By default np.arange does not account for the upper bound
    We have modified here to produce a linspace similar to Matlab

    Parameters
    ----------
    a : float
        Lower bound
    b : float
        Upper bound
    step : float
        Step size

    Returns
    -------
    The array of linspace

    """
    
    b += (lambda x: step*max(0.1, x) if x < 0.5 else 0)((lambda n: n-int(n))((b - a)/step+1))
    
    return np.arange(a, b, step)