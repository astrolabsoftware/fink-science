# Author: Alexandre Boucaud and Emille E. O. Ishida
#         Based on initial prototype developed by the CRP #4 team
#
# created on 25 January 2018
#
# Licensed GNU General Public License v3.0;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from scipy.optimize import least_squares

def bazin(time, a, b, t0, tfall, trise):
    """ Parametric light curve function proposed by Bazin et al., 2009.

    Parameters
    ----------
    time : np.array
        exploratory variable (time of observation)
    a: float
        Normalization parameter
    b: float
        Shift parameter
    t0: float
        Time of maximum
    tfall: float
        Characteristic decline time
    trise: float
        Characteristic raise time

    Returns
    -------
    array_like
        response variable (flux)
    """
    X = np.exp(-(time - t0) / tfall) / (1 + np.exp((time - t0) / trise))

    return a * X + b


def errfunc(params, time, flux):
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

    return abs(flux - bazin(time, *params))


def fit_scipy(time, flux):
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
    time = time - time[0]
    flux = np.asarray(flux)
    t0 = time[flux.argmax()] - time[0]
    guess = [0, 0, t0, 40, -5]
    # guess = [1, np.mean(flux), t0, 40, -5]
    # guess = [np.mean(flux), 0, t0, 40, -5]

    result = least_squares(errfunc, guess, args=(time, flux), method='lm')

    # Check for failures (NaNs)
    if sum([item != item for item in result.x]) > 0:
        print('fit fails')
        result.x = np.zeros(5, dtype=np.float)

    return result.x
