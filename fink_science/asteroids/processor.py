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
from pyspark.sql.functions import pandas_udf

from fink_science import __file__
import os

import pandas as pd
import numpy as np

from pyspark.sql.types import (
    IntegerType,
    ArrayType,
    FloatType,
    StructType,
    StructField,
    StringType,
)

from fink_science.tester import spark_unit_tests
from fink_fat.streaming_associations.fink_fat_associations import fink_fat_association


roid_schema = StructType(
    [
        StructField(
            "roid",
            IntegerType(),
            True,
        ),
        StructField(
            "ffdistnr",
            ArrayType(FloatType()),
            True,
        ),
        StructField(
            "estimator_id",
            ArrayType(StringType()),
            True,
        ),
    ]
)


@pandas_udf(roid_schema)
def roid_catcher(
    ra,
    dec,
    jd,
    magpsf,
    candid,
    cjd,
    cmagpsf,
    fid,
    ndethist,
    sgscore1,
    ssdistnr,
    distpsnr1,
    error_radius,
    mag_criterion_same_fid,
    mag_criterion_diff_fid,
    orbit_tw,
    orbit_error,
    confirmed_sso,
):
    """Determine if an alert is a potential Solar System object (SSO) using two criteria:

    1. The alert has been flagged as an SSO by ZTF (MPC) within 5"
    2. The alert satisfies Fink criteria for a SSO
        1. No stellar counterpart from PS1, sgscore1 < 0.76 (Tachibana & Miller 2018)
        2. Number of detection is 1 or 2
        3. No Panstarrs counterpart within 1"
        4. If 2 detections, observations must be done within 30 min.

    The alerts are labeled using:
        * [5] if the alert has been associated with a candidate trajectory using an orbit estimator
        * [4] if the alert has been associated with a candidate trajectory using a polyfit estimator
        * [3] if the alert has been flagged by ZTF as SSO candidate
        * [2] if the alert has been flagged by Fink as SSO candidate
        * [1] if is the first time ZTF sees this object
        * [0] if it is likely not a SSO

    Parameters
    ----------
    ra: Spark DataFrame Column
        right ascension
    dec: Spark DataFrame Column
        declination
    jd: Spark DataFrame Column
        Observation Julian date at start of exposure [days]
    magpsf: Spark DataFrame Column
        Magnitude from PSF-fit photometry [mag]
    candid: Spark DataFrame Column
        alert identifier
    cjd : Spark DataFrame Column
        julian date history of the alerts
    cmagpsf : Spark DataFrame Column
        magnitude history of the alerts
    fid: Spark DataFrame Column
        filter identifier (for ZTF, 1 = g band and 2 = r band)
    ndethist: Spark DataFrame Column
        Number of spatially-coincident detections falling within 1.5 arcsec
        going back to beginning of survey; only detections that fell on the
        same field and readout-channel ID where the input candidate was
        observed are counted. All raw detections down to a
        photometric S/N of ~ 3 are included. [int]
    sgscore1: Spark DataFrame Column
        Star/Galaxy score of closest source from PS1 catalog
        0 <= sgscore <= 1 where closer to 1 implies higher
        likelihood of being a star [float]
    ssdistnr: Spark DataFrame Column
        Distance to nearest known solar system object,
        set to -999.0 if none [arcsec].
    distpsnr1: Spark DataFrame Column
        Distance of closest source from PS1 catalog;
        if exists within 30 arcsec [arcsec]
    error_radius: Spark DataFrame Column
        error radius used to associates the alerts with a candidate trajectory
    mag_criterion_same_fid: Spark DataFrame Column
        keep the association where the difference of magnitude between two measurements of the
        same filter are below this threshold.
    mag_criterion_diff_fid: Spark DataFrame Column
        keep the association where the difference of magnitude
        between two measurements of differents filter are below this threshold.
    orbit_tw : int
        time window used to filter the orbit
    orbit_error: float
        error radius to associates the alerts with the orbits
    confirmed_sso: Spark DataFrame Column
        if true, associates alerts with a flag equals to 3,
        choose alerts with a flag equals to 1 or 2 otherwise.

    Returns
    ----------
    roid: integer
        5 if the alert has been associated with a candidate trajectory using an orbit estimator
        4 if the alert has been associated with a candidate trajectory using a polyfit estimator
        3 if the alert has been flagged by ZTF as SSO
        2 if the alert has been flagged by Fink as SSO
        1 if it is the first time ZTF sees this object
        0 if it is likely not a SSO
    ffdistnr : float list
        distance from the trajectory prediction
            - in arcmin if flag == 4
            - in arcsecond if flag == 5
    estimator_id: string list
        The fink_fat trajectory identifier associated with the alerts (only if roid is 4 or 5)
            - Is a integer if associated with a trajectory candidate
            - is a string if associated with an orbit

    Examples
    ----------
    >>> from fink_utils.spark.utils import concat_col
    >>> from pyspark.sql import functions as F
    >>> from fink_science.tester import add_roid_datatest
    >>> add_roid_datatest(spark, True)

    >>> df = spark.read.load(ztf_alert_sample)

    # Required alert columns
    >>> what = ['jd', 'magpsf']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    # Perform the fit + classification (default model)
    >>> args = [
    ...     'candidate.ra', 'candidate.dec',
    ...     'candidate.jd', 'candidate.magpsf',
    ...     'candidate.candid',
    ...     'cjd', 'cmagpsf',
    ...     'candidate.fid',
    ...     'candidate.ndethist', 'candidate.sgscore1',
    ...     'candidate.ssdistnr', 'candidate.distpsnr1',
    ...     F.lit(2), F.lit(2), F.lit(30), F.lit(15.0), F.lit(True)
    ... ]
    >>> df = df.withColumn('roid', roid_catcher(*args))

    # Drop temp columns
    >>> df = df.drop(*what_prefix)

    >>> df.filter(df['roid.roid'] == 2).count()
    175
    >>> df.filter(df['roid.roid'] == 3).count()
    6694
    >>> df.filter(df['roid.roid'] == 4).count()
    2
    >>> df.filter(df['roid.roid'] == 5).count()
    3
    """
    flags = np.zeros_like(ndethist.values, dtype=int)

    # remove NaN
    nalerthist = cmagpsf.apply(lambda x: np.sum(np.array(x) == np.array(x)))

    # first detection
    f0 = ndethist == 1
    flags[f0] = 1

    # Probable asteroid
    f1 = sgscore1 < 0.76
    f2 = ndethist <= 2
    flags[f1 & f2] = 2

    # criterion on distance to Panstarr (1 arcsec)
    f_distance1 = distpsnr1 < 5
    f_distance2 = distpsnr1 > 0
    mask_roid = f_distance1 & f_distance2
    flags[mask_roid] = 0

    # Remove long trend (within the observation)
    f3 = nalerthist == 2
    f4 = cjd[f3].apply(lambda x: np.diff(x)[-1]) > (30.0 / (24.0 * 60.0))
    flags[f3 & f4] = 0

    # Remove very long trend (outside the current observation)
    f5 = (ndethist == 2) & (nalerthist == 1)
    flags[f5] = 0

    # Add alerts caught by ZTF already from MPC
    if ssdistnr is not None:
        # alerts should be at max 5'' away from MPC object
        f_distance1 = ssdistnr >= 0.0
        f_distance2 = ssdistnr < 5.0

        # Distance to Panstarrs object should be bigger than distance to MPC object
        f_relative_distance = (abs(distpsnr1) - ssdistnr) > 0.0

        # Not seen many times with the same objectId
        f_ndethist = ndethist <= 5
        f_nalerthist = nalerthist <= 5

        mask_roid = (
            f_distance1 & f_distance2 & f_relative_distance & f_ndethist & f_nalerthist
        )
        flags[mask_roid] = 3

    # fink_fat associations
    flags, estimator_id, ffdistnr = fink_fat_association(
        ra,
        dec,
        magpsf,
        fid,
        jd,
        candid,
        flags,
        confirmed_sso,
        error_radius,
        mag_criterion_same_fid,
        mag_criterion_diff_fid,
        orbit_tw,
        orbit_error,
    )

    return pd.DataFrame(
        {
            "roid": flags,
            "ffdistnr": ffdistnr,
            "estimator_id": estimator_id,
        }
    )


if __name__ == "__main__":
    """Execute the test suite"""

    globs = globals()
    path = os.path.dirname(__file__)
    ztf_alert_sample = "file://{}/data/alerts/roid_datatest/alerts_sample_roid".format(
        path
    )
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
