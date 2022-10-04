# Copyright 2019-2022 AstroLab Software
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
from pyspark.sql.types import DoubleType, StringType

import pandas as pd
import numpy as np

import os

from fink_science import __file__

from fink_utils.data.utils import format_data_as_snana
from fink_utils.data.utils import load_scikit_model
from fink_utils.xmatch.simbad import return_list_of_eg_host

from actsnfink.classifier_sigmoid import get_sigmoid_features_dev
from actsnfink.classifier_sigmoid import get_sigmoid_features_elasticc

from actsnfink.classifier_sigmoid import RF_FEATURE_NAMES

from fink_science.tester import spark_unit_tests

def apply_selection_cuts_ztf(
        magpsf: pd.Series, ndethist: pd.Series, cdsxmatch: pd.Series,
        minpoints: int = 4, maxndethist: int = 20) -> pd.Series:
    """ Apply selection cuts to keep only alerts of interest
    for early SN Ia analysis

    Parameters
    ----------
    magpsf: pd.Series
        Series containing data measurement (array of double). Each row contains
        all measurement values for one alert.
    ndethist: pd.Series
        Series containing length of the alert history (int).
        Each row contains the (single) length of the alert.
    cdsxmatch: pd.Series
        Series containing crossmatch label with SIMBAD (str).
        Each row contains one label.

    Returns
    ---------
    mask: pd.Series
        Series containing `True` if the alert is valid, `False` otherwise.
        Each row contains one boolean.
    """
    # Flag alerts with less than 3 points in total
    mask = magpsf.apply(lambda x: np.sum(np.array(x) == np.array(x))) >= minpoints

    # only alerts with less or equal than 20 measurements
    mask *= (ndethist.astype(int) <= maxndethist)

    # reject galactic objects
    list_of_sn_host = return_list_of_eg_host()
    mask *= cdsxmatch.apply(lambda x: x in list_of_sn_host)

    return mask

@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def rfscore_sigmoid_full(jd, fid, magpsf, sigmapsf, cdsxmatch, ndethist, model=None) -> pd.Series:
    """ Return the probability of an alert to be a SNe Ia using a Random
    Forest Classifier (sigmoid fit).

    You need to run the SIMBAD crossmatch before.

    Parameters
    ----------
    jd: Spark DataFrame Column
        JD times (vectors of floats)
    fid: Spark DataFrame Column
        Filter IDs (vectors of ints)
    magpsf, sigmapsf: Spark DataFrame Columns
        Magnitude from PSF-fit photometry, and 1-sigma error (vectors of floats)
    cdsxmatch: Spark DataFrame Column
        Type of object found in Simbad (string)
    ndethist: Spark DataFrame Column
        Column containing the number of detection by ZTF at 3 sigma (int)
    model: Spark DataFrame Column, optional
        Path to the trained model. Default is None, in which case the default
        model `data/models/default-model.obj` is loaded.

    Returns
    ----------
    probabilities: 1D np.array of float
        Probability between 0 (non-Ia) and 1 (Ia).

    Examples
    ----------
    >>> from fink_science.xmatch.processor import xmatch_cds
    >>> from fink_utils.spark.utils import concat_col
    >>> from pyspark.sql import functions as F

    >>> df = spark.read.load(ztf_alert_sample)

    >>> df = xmatch_cds(df)

    # Required alert columns
    >>> what = ['jd', 'fid', 'magpsf', 'sigmapsf']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    # Perform the fit + classification (default model)
    >>> args = [F.col(i) for i in what_prefix]
    >>> args += [F.col('cdsxmatch'), F.col('candidate.ndethist')]
    >>> df = df.withColumn('pIa', rfscore_sigmoid_full(*args))

    >>> df.filter(df['pIa'] > 0.5).count()
    6

    >>> df.filter(df['pIa'] > 0.5).select(['rf_snia_vs_nonia', 'pIa']).show()
    +----------------+-----+
    |rf_snia_vs_nonia|  pIa|
    +----------------+-----+
    |           0.839|0.689|
    |           0.782|0.649|
    |           0.887|0.803|
    |           0.785|0.674|
    |            0.88|0.816|
    |           0.777|0.693|
    +----------------+-----+
    <BLANKLINE>

    # Note that we can also specify a model
    >>> args = [F.col(i) for i in what_prefix]
    >>> args += [F.col('cdsxmatch'), F.col('candidate.ndethist')]
    >>> args += [F.lit(model_path_sigmoid)]
    >>> df = df.withColumn('pIa', rfscore_sigmoid_full(*args))

    >>> df.filter(df['pIa'] > 0.5).count()
    6

    >>> df.agg({"pIa": "max"}).collect()[0][0] < 1.0
    True
    """
    mask = apply_selection_cuts_ztf(magpsf, ndethist, cdsxmatch)

    if len(jd[mask]) == 0:
        return pd.Series(np.zeros(len(jd), dtype=float))

    candid = pd.Series(range(len(jd)))
    pdf = format_data_as_snana(jd, magpsf, sigmapsf, fid, candid, mask)

    # Load pre-trained model `clf`
    if model is not None:
        clf = load_scikit_model(model.values[0])
    else:
        curdir = os.path.dirname(os.path.abspath(__file__))
        model = curdir + '/data/models/default-model_sigmoid.obj'
        clf = load_scikit_model(model)

    test_features = []
    flag = []
    for id in np.unique(pdf['SNID']):
        pdf_sub = pdf[pdf['SNID'] == id]
        features = get_sigmoid_features_dev(pdf_sub)
        if (features[0] == 0) or (features[6] == 0):
            flag.append(False)
        else:
            flag.append(True)
        test_features.append(features)

    flag = np.array(flag, dtype=np.bool)

    # Make predictions
    probabilities = clf.predict_proba(test_features)
    probabilities[~flag] = 0.0

    # Take only probabilities to be Ia
    to_return = np.zeros(len(jd), dtype=float)
    to_return[mask] = probabilities.T[1]

    return pd.Series(to_return)

@pandas_udf(StringType(), PandasUDFType.SCALAR)
def extract_features_rf_snia(jd, fid, magpsf, sigmapsf, cdsxmatch, ndethist) -> pd.Series:
    """ Return the features used by the RF classifier.

    There are 12 features. Order is:
    a_g,b_g,c_g,snratio_g,chisq_g,nrise_g,
    a_r,b_r,c_r,snratio_r,chisq_r,nrise_r

    Parameters
    ----------
    jd: Spark DataFrame Column
        JD times (float)
    fid: Spark DataFrame Column
        Filter IDs (int)
    magpsf, sigmapsf: Spark DataFrame Columns
        Magnitude from PSF-fit photometry, and 1-sigma error
    cdsxmatch: Spark DataFrame Column
        Type of object found in Simbad (string)
    ndethist: Spark DataFrame Column
        Column containing the number of detection by ZTF at 3 sigma (int)

    Returns
    ----------
    features: list of str
        List of string.

    Examples
    ----------
    >>> from pyspark.sql.functions import split
    >>> from pyspark.sql.types import FloatType
    >>> from fink_utils.spark.utils import concat_col
    >>> from pyspark.sql import functions as F

    >>> df = spark.read.load(ztf_alert_sample)

    # Required alert columns
    >>> what = ['jd', 'fid', 'magpsf', 'sigmapsf']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    # Perform the fit + classification (default model)
    >>> args = [F.col(i) for i in what_prefix]
    >>> args += [F.col('cdsxmatch'), F.col('candidate.ndethist')]
    >>> df = df.withColumn('features', extract_features_rf_snia(*args))

    >>> for name in RF_FEATURE_NAMES:
    ...   index = RF_FEATURE_NAMES.index(name)
    ...   df = df.withColumn(name, split(df['features'], ',')[index].astype(FloatType()))

    # Trigger something
    >>> df.agg({RF_FEATURE_NAMES[0]: "min"}).collect()[0][0]
    0.0
    """
    mask = apply_selection_cuts_ztf(magpsf, ndethist, cdsxmatch)

    if len(jd[mask]) == 0:
        return pd.Series(np.zeros(len(jd), dtype=float))

    candid = pd.Series(range(len(jd)))
    pdf = format_data_as_snana(jd, magpsf, sigmapsf, fid, candid, mask)

    test_features = []
    for id in np.unique(pdf['SNID']):
        pdf_sub = pdf[pdf['SNID'] == id]
        features = get_sigmoid_features_dev(pdf_sub)
        test_features.append(features)

    to_return_features = np.zeros((len(jd), len(RF_FEATURE_NAMES)), dtype=float)
    to_return_features[mask] = test_features

    concatenated_features = [
        ','.join(np.array(i, dtype=str)) for i in to_return_features
    ]

    return pd.Series(concatenated_features)

@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def rfscore_sigmoid_elasticc(midPointTai, filterName, psFlux, psFluxErr, cdsxmatch, nobs, maxduration=None, model=None) -> pd.Series:
    """ Return the probability of an alert to be a SNe Ia using a Random
    Forest Classifier (sigmoid fit) on ELaSTICC alert data.

    You need to run the SIMBAD crossmatch before.

    Parameters
    ----------
    midPointTai: Spark DataFrame Column
        JD times (vectors of floats)
    filterName: Spark DataFrame Column
        Filter IDs (vectors of str)
    psFlux, psFluxErr: Spark DataFrame Columns
        SNANA calibrated flux, and 1-sigma error (vectors of floats)
    cdsxmatch: Spark DataFrame Column
        Type of object found in Simbad (string)
    nobs: Spark DataFrame Column
        Column containing the number of detections by LSST
    maxduration: Spark DataFrame Column
        Integer for the maximum duration (in days) of the lightcurve to be classified.
        Default is None, i.e. no maximum duration
    model: Spark DataFrame Column, optional
        Path to the trained model. Default is None, in which case the default
        model `data/models/default-model.obj` is loaded.

    Returns
    ----------
    probabilities: 1D np.array of float
        Probability between 0 (non-Ia) and 1 (Ia).

    Examples
    ----------
    >>> from fink_utils.spark.utils import concat_col
    >>> from pyspark.sql import functions as F

    >>> df = spark.read.format('parquet').load(elasticc_alert_sample)

    # Assuming random positions
    >>> df = df.withColumn('cdsxmatch', F.lit('Unknown'))

    # Required alert columns
    >>> what = ['midPointTai', 'filterName', 'psFlux', 'psFluxErr']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...     df = concat_col(
    ...         df, colname, prefix=prefix,
    ...         current='diaSource', history='prvDiaSources')

    # Perform the fit + classification (default model)
    >>> args = [F.col(i) for i in what_prefix]
    >>> args += [F.col('cdsxmatch'), F.col('diaSource.nobs')]
    >>> df = df.withColumn('pIa', rfscore_sigmoid_elasticc(*args))

    >>> df.filter(df['pIa'] > 0.5).count()
    0
    """
    mask = apply_selection_cuts_ztf(psFlux, nobs, cdsxmatch, maxndethist=100)

    dt = midPointTai.apply(lambda x: np.max(x) - np.min(x))

    # Maximum days in the history
    if maxduration is not None:
        mask *= (dt <= maxduration.values[0])

    if len(midPointTai[mask]) == 0:
        return pd.Series(np.zeros(len(midPointTai), dtype=float))

    candid = pd.Series(range(len(midPointTai)))
    pdf = format_data_as_snana(
        midPointTai, psFlux, psFluxErr,
        filterName, candid, mask,
        transform_to_flux=False
    )

    # Load pre-trained model `clf`
    if model is not None:
        clf = load_scikit_model(model.values[0])
    else:
        curdir = os.path.dirname(os.path.abspath(__file__))
        model = curdir + '/data/models/default-model_sigmoid_elasticc.obj'
        clf = load_scikit_model(model)

    test_features = []
    flag = []
    for id in np.unique(pdf['SNID']):
        f1 = pdf['SNID'] == id
        pdf_sub = pdf[f1]
        features = get_sigmoid_features_elasticc(pdf_sub)

        # Do not classify if less than 2 bands
        feats = []
        nfeat_per_band = 6
        nbands = 6
        for i in range(nbands):
            feats.append(features[i * nfeat_per_band])

        n_nonzero_feats = np.sum(np.array(feats) != 0)

        if n_nonzero_feats < 2:
            flag.append(False)
        else:
            flag.append(True)
        test_features.append(features)

    flag = np.array(flag, dtype=np.bool)

    # Make predictions
    probabilities = clf.predict_proba(test_features)
    probabilities[~flag] = 0.0

    # Take only probabilities to be Ia
    to_return = np.zeros(len(midPointTai), dtype=float)
    to_return[mask] = probabilities.T[1]

    return pd.Series(to_return)


if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = 'file://{}/data/alerts/datatest'.format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    elasticc_alert_sample = 'file://{}/data/alerts/elasticc_parquet'.format(path)
    globs["elasticc_alert_sample"] = elasticc_alert_sample

    model_path_sigmoid = '{}/data/models/default-model_sigmoid.obj'.format(path)
    globs["model_path_sigmoid"] = model_path_sigmoid

    # Run the test suite
    spark_unit_tests(globs)
