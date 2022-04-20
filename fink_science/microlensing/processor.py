# Copyright 2020-2022 AstroLab Software
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
from pyspark.sql.types import StringType, DoubleType

import numpy as np
import pandas as pd

import os
import warnings

from fink_science import __file__
from fink_science.microlensing.classifier import _extract
from fink_science.microlensing.classifier import LIA_FEATURE_NAMES
from fink_science.microlensing.classifier import load_external_model

from fink_utils.photometry.conversion import dc_mag

from LIA import microlensing_classifier

from fink_science.tester import spark_unit_tests

@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def mulens(
        fid, magpsf, sigmapsf, magnr, sigmagnr,
        magzpsci, isdiffpos, ndethist):
    """ Returns the predicted class (among microlensing, variable star,
    cataclysmic event, and constant event) & probability of an alert to be
    a microlensing event in each band using a Random Forest Classifier.

    Parameters
    ----------
    fid: Spark DataFrame Column
        Filter IDs (int)
    magpsf, sigmapsf: Spark DataFrame Columns
        Magnitude from PSF-fit photometry, and 1-sigma error
    magnr, sigmagnr: Spark DataFrame Columns
        Magnitude of nearest source in reference image PSF-catalog
        within 30 arcsec and 1-sigma error
    magzpsci: Spark DataFrame Column
        Magnitude zero point for photometry estimates
    isdiffpos: Spark DataFrame Column
        t => candidate is from positive (sci minus ref) subtraction
        f => candidate is from negative (ref minus sci) subtraction

    Returns
    ----------
    out: list
        Returns the mean of the probabilities (one probability per band) if the
        event was considered as microlensing in both bands, otherwise 0.0.

    Examples
    ---------
    >>> from fink_utils.spark.utils import concat_col
    >>> from pyspark.sql import functions as F

    >>> df = spark.read.load(ztf_alert_sample)

    # Required alert columns
    >>> what = [
    ...    'fid', 'magpsf', 'sigmapsf',
    ...    'magnr', 'sigmagnr', 'magzpsci', 'isdiffpos']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    >>> args = [F.col(i) for i in what_prefix]
    >>> args += ['candidate.ndethist']
    >>> df = df.withColumn('new_mulens', mulens(*args))

    # Drop temp columns
    >>> df = df.drop(*what_prefix)

    >>> df.filter(df['new_mulens'] > 0.0).count()
    0
    """
    warnings.filterwarnings('ignore')

    # broadcast models
    curdir = os.path.dirname(os.path.abspath(__file__))
    model_path = curdir + '/data/models/'
    rf, pca = load_external_model(model_path)

    valid_index = np.arange(len(magpsf), dtype=int)

    # At most 100 measurements in each band
    mask = (ndethist.astype(int) < 100)

    # At least 10 measurements in each band
    mask *= magpsf.apply(lambda x: np.sum(np.array(x) == np.array(x))) >= 20

    to_return = np.zeros(len(magpsf), dtype=float)

    for index in valid_index[mask.values]:
        # Select only valid measurements (not upper limits)
        maskNotNone = np.array(magpsf.values[index]) == np.array(magpsf.values[index])

        classes = []
        probs = []
        for filt in [1, 2]:
            maskFilter = np.array(fid.values[index]) == filt
            m = maskNotNone * maskFilter

            # Reject if less than 10 measurements
            if np.sum(m) < 10:
                classes.append('')
                continue

            # Compute DC mag
            mag, err = np.array([
                dc_mag(i[0], i[1], i[2], i[3], i[4], i[5], i[6])
                for i in zip(
                    np.array(fid.values[index])[m],
                    np.array(magpsf.values[index])[m],
                    np.array(sigmapsf.values[index])[m],
                    np.array(magnr.values[index])[m],
                    np.array(sigmagnr.values[index])[m],
                    np.array(magzpsci.values[index])[m],
                    np.array(isdiffpos.values[index])[m])
            ]).T

            # Run the classifier
            output = microlensing_classifier.predict(mag, err, rf, pca)

            # Update the results
            # Beware, in the branch FINK the order has changed
            # classification,p_cons,p_CV,p_ML,p_var = microlensing_classifier.predict()
            classes.append(str(output[0]))
            probs.append(float(output[3][0]))

        # Append mean of classification if ML favoured, otherwise 0
        if np.all(np.array(classes) == 'ML'):
            to_return[index] = np.mean(probs)
        else:
            to_return[index] = 0.0

    return pd.Series(to_return)

@pandas_udf(StringType(), PandasUDFType.SCALAR)
def extract_features_mulens(
        fid, magpsf, sigmapsf, magnr, sigmagnr,
        magzpsci, isdiffpos):
    """ Returns the predicted class (among microlensing, variable star,
    cataclysmic event, and constant event) & probability of an alert to be
    a microlensing event in each band using a Random Forest Classifier.

    Parameters
    ----------
    fid: Spark DataFrame Column
        Filter IDs (int)
    magpsf, sigmapsf: Spark DataFrame Columns
        Magnitude from PSF-fit photometry, and 1-sigma error
    magnr, sigmagnr: Spark DataFrame Columns
        Magnitude of nearest source in reference image PSF-catalog
        within 30 arcsec and 1-sigma error
    magzpsci: Spark DataFrame Column
        Magnitude zero point for photometry estimates
    isdiffpos: Spark DataFrame Column
        t => candidate is from positive (sci minus ref) subtraction
        f => candidate is from negative (ref minus sci) subtraction

    Returns
    ----------
    out: list of string
        Return the features (2 * 47)

    Examples
    ----------
    >>> from pyspark.sql.functions import split
    >>> from pyspark.sql.types import FloatType
    >>> from fink_utils.spark.utils import concat_col
    >>> from pyspark.sql import functions as F

    >>> df = spark.read.load(ztf_alert_sample)

    # Required alert columns
    >>> what = ['fid', 'magpsf', 'sigmapsf', 'magnr', 'sigmagnr', 'magzpsci', 'isdiffpos']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    # Perform the fit + classification (default model)
    >>> args = [F.col(i) for i in what_prefix]
    >>> df = df.withColumn('features', extract_features_mulens(*args))

    >>> for name in LIA_FEATURE_NAMES:
    ...   index = LIA_FEATURE_NAMES.index(name)
    ...   df = df.withColumn(name, split(df['features'], ',')[index].astype(FloatType()))

    # Trigger something
    >>> df.agg({LIA_FEATURE_NAMES[0]: "min"}).collect()[0][0]
    0.0
    """
    warnings.filterwarnings('ignore')

    # Loop over alerts
    outs = []
    for index in range(len(fid)):
        # Select only valid measurements (not upper limits)
        maskNotNone = np.array(magpsf.values[index]) == np.array(magpsf.values[index])

        # Loop over filters
        out = ''
        for filt in [1, 2]:
            maskFilter = np.array(fid.values[index]) == filt
            m = maskNotNone * maskFilter

            # Reject if less than 10 measurements
            if np.sum(m) < 10:
                out += ','.join(['0'] * len(LIA_FEATURE_NAMES))
                continue

            # Compute DC mag
            mag, err = np.array([
                dc_mag(i[0], i[1], i[2], i[3], i[4], i[5], i[6])
                for i in zip(
                    np.array(fid.values[index])[m],
                    np.array(magpsf.values[index])[m],
                    np.array(sigmapsf.values[index])[m],
                    np.array(magnr.values[index])[m],
                    np.array(sigmagnr.values[index])[m],
                    np.array(magzpsci.values[index])[m],
                    np.array(isdiffpos.values[index])[m])
            ]).T

            # Run the classifier
            output = _extract(mag, err)

            # Update the results
            out += output
        outs.append(out)

    return pd.Series(outs)


if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    path = os.path.dirname(__file__)
    ztf_alert_sample = 'file://{}/data/alerts/datatest'.format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    model_path = '{}/data/models'.format(path)
    globs["model_path"] = model_path

    # Run the test suite
    spark_unit_tests(globs)
