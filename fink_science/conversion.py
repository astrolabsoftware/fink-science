# Copyright 2020 AstroLab Software
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
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import DoubleType

from fink_science.tester import spark_unit_tests

import numpy as np
import pandas as pd

@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def magpsf2dcmag(lfid, lmagpsf, lmagnr, lmagzpsci, lisdiffpos):
    """ Compute DC mag from difference magnitude supplied by ZTF.

    This was heavily influenced by the computation provided by Lasair:
    https://github.com/lsst-uk/lasair/blob/master/src/alert_stream_ztf/common/mag.py

    Parameters
    ----------
    lfid: int
        filter, 1 for green and 2 for red
    lmagpsf: 1D arrays of float
        magnitude from PSF-fit photometry
    lmagnr: 1D arrays of float
        magnitude of nearest source in reference image PSF-catalog
        within 30 arcsec
    lmagzpsci: 1D array of float
        Magnitude zero point for photometry estimates
    lisdiffpos: 1D array of string
        t => candidate is from positive (sci minus ref) subtraction
        f => candidate is from negative (ref minus sci) subtraction

    Returns
    ---------
    dc_mag: apparent magnitude.

    Examples
    ----------
    >>> from fink_science.random_forest_snia.classifier import concat_col
    >>> from pyspark.sql import functions as F

    >>> df = spark.read.load(ztf_alert_sample)

    # Required alert columns
    >>> what = ['fid', 'magpsf', 'magnr', 'magzpsci', 'isdiffpos']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    # Perform the fit + classification (default model)
    >>> args = [F.col(i) for i in what_prefix]
    >>> df = df.withColumn('dc_mag', magpsf2dcmag(*args))

    >>> df.show()
    """
    for alert in zip(fid, magpsf, magnr, magzpsci, isdiffpos):
        fid, magpsf, magnr, magzpsci, isdiffpos = alert
        # Design zero points.
        ref_zps = {1: 26.325, 2: 26.275, 3: 25.660}
        magzpref = np.array([
            np.array([ref_zps[int(i)] for i in alert_row])
            for alert_row in fid.values
        ])

        # reference flux and its error
        magdiff = magzpref - magnr
        mask = np.array([i > 12 for i in magdiff])
        magdiff[mask] = 12.0
        ref_flux = 10**(0.4 * (magdiff))

        # difference flux and its error
        mask = magzpsci == 0
        magzpsci[mask] = magzpref[mask]
        magdiff = magzpsci - magpsf
        mask = magdiff > 12
        magdiff[mask] = 12.0
        difference_flux = 10**(0.4 * (magdiff))

        # add or subract difference flux based on isdiffpos
        mask = isdiffpos == 't'
        dc_flux = np.zeros_like(ref_flux)
        dc_flux[mask] = ref_flux[mask] + difference_flux[mask]
        dc_flux[~mask] = ref_flux[~mask] - difference_flux[~mask]

        # apparent mag
        if (dc_flux == dc_flux) and dc_flux > 0.0:
            dc_mag = magzpsci - 2.5 * np.log10(dc_flux)
        else:
            dc_mag = magzpsci

    return pd.Series(dc_mags)

@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def sigmapsf2dcerr(fid, magpsf, sigmapsf, magnr, sigmagnr, magzpsci, isdiffpos):
    """ Compute DC mag error from difference magnitude supplied by ZTF.

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
    dc_sigmag: apparent magnitude error
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

    # apparent mag and its error from fluxes
    if (dc_flux == dc_flux) and dc_flux > 0.0:
        dc_sigmag = dc_sigflux / dc_flux * 1.0857
    else:
        dc_sigmag = sigmapsf

    return pd.Series(dc_sigmag)

def mag2flux(mag, magzpsci, magerr) -> (list, list):
    """ Convert DC mag and error into flux and flux error

    Parameters
    ----------
    mag: 1D array of float
        Array or apparent magnitudes
    magzpsci: 1D array of float
        Magnitude zero point for photometry estimates
    magerr: 1D array of float
        Array or apparent magnitude errors
    """
    flux = 10 ** ((magzpsci - mag) / 2.5)
    flux_err = magerr * flux / 1.0857

    return flux, flux_err


if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    ztf_alert_sample = 'fink_science/data/alerts/alerts.parquet'
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
