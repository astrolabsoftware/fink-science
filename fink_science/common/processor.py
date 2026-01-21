# Copyright 2022 AstroLab Software
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
from pyspark.sql.types import ArrayType, FloatType

import pandas as pd
import numpy as np

import os

from fink_science import __file__
from fink_utils.photometry.vect_conversion import vect_dc_mag

from fink_science.tester import spark_unit_tests

@pandas_udf(ArrayType(ArrayType(FloatType())), PandasUDFType.SCALAR)
def dcmag(fid, magpsf, sigmapsf, magnr, sigmagnr, magzpsci, isdiffpos):
    """ Correct lightcurves (DC magnitude)

    Parameters
    ----------
    fid: ints
        filter, 1 for green and 2 for red
    magpsf, sigmapsf; floats
        magnitude from PSF-fit photometry, and 1-sigma error
    magnr, sigmagnr: floats
        magnitude of nearest source in reference image PSF-catalog
        within 30 arcsec and 1-sigma error
    magzpsci: floats
        Magnitude zero point for photometry estimates
    isdiffpos: strs
        t or 1 => candidate is from positive (sci minus ref) subtraction;
        f or 0 => candidate is from negative (ref minus sci) subtraction

    Returns
    ----------
    pd.Series
        Series with [dcmag, dcmagerr] rows

    Examples
    ----------
    >>> from fink_science.xmatch.processor import xmatch_cds
    >>> from fink_utils.spark.utils import concat_col
    >>> from pyspark.sql import functions as F

    >>> df = spark.read.load(ztf_alert_sample)

    >>> df = xmatch_cds(df)

    # Required alert columns
    >>> what = ['jd', 'fid', 'magpsf', 'sigmapsf', 'magnr', 'sigmagnr', 'magzpsci', 'isdiffpos']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    >>> df = df.withColumn(
    ...     'dc',
    ...     dcmag('cfid', 'cmagpsf', 'csigmapsf', 'cmagnr', 'csigmagnr', 'cmagzpsci', 'cisdiffpos')
    ... ).withColumn(
    ...     'dcmag',
    ...     F.col('dc').getItem(0)
    ... ).withColumn(
    ...     'dcmagerr',
    ...     F.col('dc').getItem(1)
    ... ).drop('dc', 'cmagpsf', 'csigmapsf', 'cmagnr', 'csigmagnr', 'cmagzpsci', 'cisdiffpos')
    """
    out = []
    for args in zip(fid, magpsf, sigmapsf, magnr, sigmagnr, magzpsci, isdiffpos):
        dcmag, err = vect_dc_mag(*args)
        out.append([dcmag, err])

    return pd.Series(out)

@pandas_udf(FloatType(), PandasUDFType.SCALAR)
def rate(magpsf, jd, fid, filt, absolute):
    """ Compute (g(t + dt) - g(t)) / (dt)
    """
    filt = filt.values[0]
    absolute = absolute.values[0]

    delta = []
    for m, t, f in zip(magpsf.values, jd.values, fid.values):
        mask = m == m

        maskfilt = f[mask] == filt

        if len(m[mask][maskfilt]) > 1:
            dm = m[mask][maskfilt][-1] - m[mask][maskfilt][-2]
            if not absolute:
                dt = t[mask][maskfilt][-1] - t[mask][maskfilt][-2]
            else:
                dt = 1.
            if dt > 0:
                delta.append(dm / dt)
            else:
                delta.append(None)
        else:
            delta.append(None)
    return pd.Series(delta)

@pandas_udf(FloatType(), PandasUDFType.SCALAR)
def color(magpsf, jd, fid, absolute):
    """ Compute (g-r)(t + dt) - (g-r)(t)
    """
    absolute = absolute.values[0]
    delta = []
    for m, t, f in zip(magpsf.values, jd.values, fid.values):
        mask = m == m

        n = len(m[mask])
        ng = np.sum(f[mask] == 1)
        nr = np.sum(f[mask] == 2)

        if (n < 4) or (ng < 2) or (nr < 2):
            delta.append(None)
        else:
            maskg = f[mask] == 1
            maskr = f[mask] == 2

            pdfg = pd.DataFrame(
                {
                    'mag': m[mask][maskg],
                    'filt': f[mask][maskg],
                    'time': t[mask][maskg]
                }
            )
            pdfr = pd.DataFrame(
                {
                    'mag': m[mask][maskr],
                    'filt': f[mask][maskr],
                    'time': t[mask][maskr]
                }
            )

            dgr0 = pdfg['mag'].values[-1] - pdfr['mag'].values[-1]
            dgr1 = pdfg['mag'].values[-2] - pdfr['mag'].values[-2]

            if absolute:
                delta.append(dgr0 - dgr1)
            else:
                t0 = (pdfg['time'].values[-1] + pdfr['time'].values[-1]) / 2
                t1 = (pdfg['time'].values[-2] + pdfr['time'].values[-2]) / 2

                dt = t0 - t1
                delta.append((dgr0 - dgr1) / dt)

    return pd.Series(delta)


if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = 'file://{}/data/alerts/datatest'.format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
