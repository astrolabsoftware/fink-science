# Copyright 2020 AstroLab Software
# Author: Marco Leoni, Julien Peloton
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
import pandas as pd

from scipy.optimize import least_squares
from scipy.stats import chisquare

def delta_t(dataframe: pd.DataFrame) -> pd.DataFrame:
    """ Re-index a DataFrame relatively to the first index.

    Parameters
    ----------
    dataframe : pandas DataFrame

    Returns
    -------
    dataframe_t0 : pandas DataFrame
        reindexed using time relative to t0
    """

    dataframe_t0 = dataframe.index - dataframe.index[0]

    return dataframe_t0


def fsigmoid(x: np.array, a: float, b: float, c: float) -> np.array:
    """Sigmoid function

    Parameters
    ---------
    x: np.array
    a: float
    b: float
    c: float

    Returns
    -------
    sigmoid: np.array
        fit with a sigmoid function
    """

    sigmoid = c / (1.0 + np.exp(-a * (x - b)))

    return np.array(sigmoid)


def errfunc_sigmoid(params: list, time: np.array, flux: np.array) -> float:
    """ Absolute difference between theoretical and measured flux.

    Parameters
    ----------
    params : list of float
        light curve parameters: (a, b, t0, tfall, trise)
    time : array_like
        exploratory variable (time of observation)
    flux : array_like
        response variable (measured flux)

    Returns
    -------
    diff : float
        absolute difference between theoretical and observed flux

    """
    return abs(flux - fsigmoid(time, *params))


def fit_sigmoid(time: np.array, flux: np.array) -> list:
    """ Find best-fit parameters using scipy.least_squares.

    Parameters
    ----------
    time : array_like
        exploratory variable (time of observation)
    flux : array_like
        response variable (measured flux)

    Returns
    -------
    output : list of float
        best fit parameter values
    """
    flux = np.asarray(flux)
    t0 = time[flux.argmax()] - time[0]
    guess = [1, t0 / 2, np.max(flux)]

    result = least_squares(errfunc_sigmoid, guess, args=(time, flux))

    return result.x


def compute_chi_square(f_obs: np.array, f_exp: np.array) -> float:
    """ Compute chisquare

    Parameters
    ----------
    f_obs: np.array
        observed data points
    f_exp: np.array
        fitted (predicted) data points

    Returns
    -------
    test_chi[0]: float
        chi_square between fitted and observed
    """

    test_chi = chisquare(f_obs, f_exp,)

    return test_chi[0]
