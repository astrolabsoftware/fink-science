# Copyright 2022 Fink Software
# Author: Etienne Russeil
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


def protected_exponent(x1):
    """Exponential function : cannot exceed e**100

    Examples
    --------
    >>> np.round_(protected_exponent(420))
    2.6881171418161356e+43
    >>> np.round_(protected_exponent(1), 5)
    2.71828
    """
    with np.errstate(over='ignore'):
        lim = 100
        protex = np.where(x1 < lim, x1, lim)
        return np.exp(protex)


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
    p1, p2, p3, p4 : floats
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
