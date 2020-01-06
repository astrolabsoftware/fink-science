# Copyright 2019 AstroLab Software
# Author: Julien Peloton
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

def mag2flux(fid, magpsf, sigmapsf, magnr, sigmagnr, magzpsci, isdiffpos):
    """ Compute flux from difference magnitude supplied by ZTF.

    This was heavily influenced by the computation provided by Lasair:
    https://github.com/lsst-uk/lasair/blob/master/src/alert_stream_ztf/common/mag.py

    Parameters
    ----------
    fid: int
        filter, 1 for green and 2 for red
    magpsf, sigmapsf: 1D arrays of float
        magnitude from PSF-fit photometry, and 1-sigma error
    magnr, sigmagnr: 1D arrays of float
        magnitude of nearest source in reference image PSF-catalog
        within 30 arcsec and 1-sigma error
    magzpsci: 1D array of float
        Magnitude zero point for photometry estimates
    isdiffpos: 1D array of string
        t => candidate is from positive (sci minus ref) subtraction
        f => candidate is from negative (ref minus sci) subtraction

    Returns
    ---------
    dc_flux, dc_sigflux: flux and its error
    """
    # Zero points. Looks like they are fixed.
    # Where this information is stored? How reliable over time it is?
    ref_zps = {1: 26.325, 2: 26.275, 3: 25.660}
    magzpref = ref_zps[fid]

    # reference flux and its error
    magdiff = magzpref - magnr
    mask = magdiff > 12
    magdiff[mask] = 12.0
    ref_flux = 10**(0.4 * (magdiff))
    ref_sigflux = (sigmagnr / 1.0857) * ref_flux

    # difference flux and its error
    mask = magzpsci == 0
    magzpsci[mask] = magzpref
    magdiff = magzpsci - magpsf
    mask = magdiff > 12
    magdiff[mask] = 12.0
    difference_flux = 10**(0.4 * (magdiff))
    difference_sigflux = (sigmapsf / 1.0857) * difference_flux

    # add or subract difference flux based on isdiffpos
    mask = isdiffpos == 't'
    dc_flux = np.zeros_like(ref_flux)
    dc_flux[mask] = ref_flux[mask] + difference_flux[mask]
    dc_flux[~mask] = ref_flux[~mask] - difference_flux[~mask]

    # assumes errors are independent. Maybe too conservative.
    dc_sigflux = np.sqrt(difference_sigflux**2 + ref_sigflux**2)

    return dc_flux, dc_sigflux
