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
from pyspark.sql.types import IntegerType

import pandas as pd
import numpy as np

from fink_science.tester import spark_unit_tests

@pandas_udf(IntegerType(), PandasUDFType.SCALAR)
def roid_catcher(fid, magpsf, ssdistnr) -> pd.Series:
    """ Determine if the alert is an asteroid using two criteria:
        1. The alert has been flagged as an asteroid by ZTF (MPC) within 5"
        2. First time the alert is seen, and |mag_r - mag_g| ~ 0.5

    The alerts are labeled using:
        [3] if the asteroid has been flagged by ZTF
        [2] if the asteroid has been flagged by Fink
        [1] if is the first time ZTF sees this object
        [0] if it is likely not an asteroid

    Parameters
    ----------
    fid: Spark DataFrame Column
        Filter IDs (ints)
    magpsf: Spark DataFrame Column
        Magnitude from PSF-fit photometry (floats)
    ssdistnr: Spark DataFrame Column
        Distance to nearest known solar system object,
        set to -999.0 if none [arcsec].

    Returns
    ----------
    out: integer
        3 if the asteroid has been flagged by ZTF
        2 if the asteroid has been flagged by Fink
        1 if it is the first time ZTF sees this object
        0 if it is likely not an asteroid

    Examples
    ----------
    >>> from fink_science.utilities import concat_col
    >>> from pyspark.sql import functions as F

    >>> df = spark.read.load(ztf_alert_sample)

    # Required alert columns
    >>> what = ['fid', 'magpsf']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    # Perform the fit + classification (default model)
    >>> args = [F.col(i) for i in what_prefix] + [F.col('candidate.ssdistnr')]
    >>> df = df.withColumn('roid', roid_catcher(*args))

    # Drop temp columns
    >>> df = df.drop(*what_prefix)

    >>> df.agg({"roid": "min"}).collect()[0][0]
    0

    >>> df.agg({"roid": "max"}).collect()[0][0]
    1
    """
    flags = []
    for mags, filts in zip(magpsf, fid):
        nmeasurements = []
        last_mags = []
        for filt in np.unique(filts):
            mask_filt = np.where(filts == filt)[0]
            mag_filt = np.array(mags)[mask_filt]
            nmeasurements.append(len([i for i in mag_filt if i == i]))
            last_mags.append(mag_filt[-1])
        if np.sum(nmeasurements) == 1:
            flags.append(1)
        elif nmeasurements == [1, 1]:
            # need to expand the logic...
            f1 = np.abs(np.diff(last_mags))[0] > 0.4
            f2 = np.abs((np.diff(last_mags)))[0] < 0.6
            if f1 & f2:
                flags.append(2)
            else:
                flags.append(0)
        else:
            flags.append(0)

    flags = np.array(flags)

    # Caught by ZTF already
    f_distance1 = ssdistnr > 0
    f_distance2 = ssdistnr < 5
    mask_roid = f_distance1 & f_distance2
    flags[mask_roid] = 3

    # return asteroid labels
    return pd.Series(flags)


if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    ztf_alert_sample = 'fink_science/data/alerts/alerts.parquet'
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
