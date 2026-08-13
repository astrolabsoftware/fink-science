# Copyright 2019-2026 AstroLab Software
# Author: Preeti Cowan, Julien Peloton
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
"""Compute the concentration of the fluxes within 2 pre-selected radii using stamps."""

import os
import io
import logging
from line_profiler import profile
import numpy as np
import pandas as pd
from photutils.aperture import CircularAperture, aperture_photometry
from astropy.io import fits

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import MapType, StringType, FloatType

from fink_science import __file__
from fink_science.tester import spark_unit_tests

_LOG = logging.getLogger(__name__)


def read_cutout_stamp(fits_bytes: bytes) -> np.ndarray:
    """
    Reads Rubin cutout stamps

    Parameters
    ----------
    fits_bytes
       input byte string
    """
    fits_buffer = io.BytesIO(fits_bytes)
    with fits.open(fits_buffer) as hdulist:
        return hdulist[0].data


def concentration(image, center, radii):
    """ """
    fluxes = []
    for r in radii:
        ap = CircularAperture(center, r)
        phot = aperture_photometry(image, ap)
        fluxes.append(phot["aperture_sum"][0])

    total_flux = np.max(fluxes)
    if total_flux <= 0:
        return np.nan

    r20 = np.interp(0.20 * total_flux, fluxes, radii)
    r80 = np.interp(0.80 * total_flux, fluxes, radii)

    if r20 <= 0:
        return np.nan
    return 5.0 * np.log10(r80 / r20)


@pandas_udf(MapType(StringType(), FloatType()))
@profile
def calculate_concentration(
    cutoutScience: pd.Series, cutoutDifference: pd.Series
) -> pd.Series:
    """Compute the concentration of the fluxes within 2 pre-selected radii using stamps.

    Notes
    -----
    Concentration index: C = 5 * log10(r80 / r20).
    Stars have high C (flux tightly concentrated).
    Comets have lower C (coma spreads the flux outward).

    Parameters
    ----------
    cutoutScience: bytes
        Science cutout
    cutoutDifference: bytes
        Difference cutout

    Returns
    -------
    out: dict
        cScience: concentration from the Science cutout
        cDifference: concentration from the Difference cutout

    Examples
    --------
    >>> df = spark.read.format('parquet').load(rubin_alert_sample)
    >>> args = ['cutoutScience', 'cutoutDifference']
    >>> df = df.withColumn('concentrations', calculate_concentration(*args))
    >>> df = df.withColumn('cScience', df['concentrations'].getItem('cScience'))
    >>> df = df.withColumn('cDifference', df['concentrations'].getItem('cDifference'))
    >>> df.select(['diaObject.diaObjectId', 'cScience', 'cDifference']).show()
    """
    radii = np.arange(1, 14)
    out = []
    for index in range(len(cutoutScience)):
        sci = read_cutout_stamp(cutoutScience.to_numpy()[index])
        dif = read_cutout_stamp(cutoutDifference.to_numpy()[index])

        # Assuming square cutouts with odd len
        center = (len(sci) // 2, len(sci) // 2)

        cScience = concentration(sci, center, radii)
        cDifference = concentration(dif, center, radii)

        out.append({"cScience": cScience, "cDifference": cDifference})

    return pd.Series(out)


if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    path = os.path.dirname(__file__)

    # from fink-alerts-schemas (see CI configuration)
    rubin_alert_sample = "file://{}/datasim/rubin_test_data_10_0.parquet".format(path)
    globs["rubin_alert_sample"] = rubin_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
