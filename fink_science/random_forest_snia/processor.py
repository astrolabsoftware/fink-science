# Copyright 2019-2020 AstroLab Software
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

import pandas as pd
import numpy as np

import os

from fink_science.conversion import mag2fluxcal_snana

from fink_science.utilities import load_scikit_model
from fink_science.random_forest_snia.classifier_bazin import fit_all_bands
from fink_science.random_forest_snia.classifier_sigmoid import get_sigmoid_features_dev

from fink_science.tester import spark_unit_tests

@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def rfscore_bazin(jd, fid, magpsf, sigmapsf, model=None) -> pd.Series:
    """ Return the probability of an alert to be a SNe Ia using a Random
    Forest Classifier (bazin fit).

    Parameters
    ----------
    jd: Spark DataFrame Column
        JD times (float)
    fid: Spark DataFrame Column
        Filter IDs (int)
    magpsf, sigmapsf: Spark DataFrame Columns
        Magnitude from PSF-fit photometry, and 1-sigma error
    model: Spark DataFrame Column, optional
        Path to the trained model. Default is None, in which case the default
        model `data/models/default-model.obj` is loaded.

    Returns
    ----------
    probabilities: 1D np.array of float
        Probability between 0 (non-Ia) and 1 (Ia).

    Examples
    ----------
    >>> from fink_science.utilities import concat_col
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
    >>> df = df.withColumn('pIa', rfscore_bazin(*args))

    # Note that we can also specify a model
    >>> args = [F.col(i) for i in what_prefix] + [F.lit(model_path_bazin)]
    >>> df = df.withColumn('pIa', rfscore_bazin(*args))

    # Drop temp columns
    >>> df = df.drop(*what_prefix)

    >>> df.agg({"pIa": "min"}).collect()[0][0]
    0.0

    >>> df.agg({"pIa": "max"}).collect()[0][0] < 1.0
    True
    """
    # Compute the test_features: fit_all_bands
    test_features = fit_all_bands(
        jd.values, fid.values, magpsf.values, sigmapsf.values)

    # Load pre-trained model `clf`
    if model is not None:
        clf = load_scikit_model(model.values[0])
    else:
        curdir = os.path.dirname(os.path.abspath(__file__))
        model = curdir + '/../data/models/default-model_bazin.obj'
        clf = load_scikit_model(model)

    # Make predictions
    probabilities = clf.predict_proba(test_features)

    # Check the type of prob
    # inverted wrt to original documentation: [pIa, pnon-Ia]
    to_return = probabilities.T[0]

    # Return probability of 0 for objects with no fit available (only zeros).
    mask = [np.sum(i) == 0.0 for i in test_features]
    to_return[mask] = 0.0

    return pd.Series(to_return)

@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def rfscore_sigmoid_full(jd, fid, magpsf, sigmapsf, model=None) -> pd.Series:
    """ Return the probability of an alert to be a SNe Ia using a Random
    Forest Classifier (sigmoid fit).

    Parameters
    ----------
    jd: Spark DataFrame Column
        JD times (float)
    fid: Spark DataFrame Column
        Filter IDs (int)
    magpsf, sigmapsf: Spark DataFrame Columns
        Magnitude from PSF-fit photometry, and 1-sigma error
    model: Spark DataFrame Column, optional
        Path to the trained model. Default is None, in which case the default
        model `data/models/default-model.obj` is loaded.

    Returns
    ----------
    probabilities: 1D np.array of float
        Probability between 0 (non-Ia) and 1 (Ia).

    Examples
    ----------
    >>> from fink_science.utilities import concat_col
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
    >>> df = df.withColumn('pIa', rfscore_sigmoid_full(*args))

    # Note that we can also specify a model
    >>> args = [F.col(i) for i in what_prefix] + [F.lit(model_path_sigmoid)]
    >>> df = df.withColumn('pIa', rfscore_sigmoid_full(*args))

    # Drop temp columns
    >>> df = df.drop(*what_prefix)

    >>> df.agg({"pIa": "min"}).collect()[0][0]
    0.0

    >>> df.agg({"pIa": "max"}).collect()[0][0] < 1.0
    True
    """
    # Flag empty alerts
    mask = magpsf.apply(lambda x: np.sum(np.array(x) == np.array(x))) > 3
    if len(jd[mask]) == 0:
        return pd.Series(np.zeros(len(jd), dtype=float))

    # add an exploded column with SNID
    df_tmp = pd.DataFrame.from_dict(
        {
            'jd': jd[mask],
            'SNID': range(len(jd[mask]))
        }
    )
    df_tmp = df_tmp.explode('jd')

    # compute flux and flux error
    data = [mag2fluxcal_snana(*args) for args in zip(
        magpsf[mask].explode(),
        sigmapsf[mask].explode())]
    flux, error = np.transpose(data)

    # make a Pandas DataFrame with exploded series
    pdf = pd.DataFrame.from_dict({
        'SNID': df_tmp['SNID'],
        'MJD': df_tmp['jd'],
        'FLUXCAL': flux,
        'FLUXCALERR': error,
        'FLT': fid[mask].explode().replace({1: 'g', 2: 'r'})
    })

    # Load pre-trained model `clf`
    if model is not None:
        clf = load_scikit_model(model.values[0])
    else:
        curdir = os.path.dirname(os.path.abspath(__file__))
        model = curdir + '/../data/models/default-model_sigmoid.obj'
        clf = load_scikit_model(model)

    test_features = []
    for id in np.unique(pdf['SNID']):
        pdf_sub = pdf[pdf['SNID'] == id]
        features = get_sigmoid_features_dev(pdf_sub)
        test_features.append(features)

    # Make predictions
    probabilities = clf.predict_proba(test_features)

    # Take only probabilities to be Ia
    to_return = np.zeros(len(jd), dtype=float)
    to_return[mask] = probabilities.T[0]

    return pd.Series(to_return)

@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def rfscore_sigmoid(
        jd, fid, magpsf, sigmapsf, model=None) -> pd.Series:
    """ Return the probability of an alert to be a SNe Ia using a Random
    Forest Classifier (sigmoid fit).

    Parameters
    ----------
    jd: Spark DataFrame Column
        JD times (float)
    fid: Spark DataFrame Column
        Filter IDs (int)
    magpsf, sigmapsf: Spark DataFrame Columns
        Magnitude from PSF-fit photometry, and 1-sigma error
    model: Spark DataFrame Column, optional
        Path to the trained model. Default is None, in which case the default
        model `data/models/default-model.obj` is loaded.

    Returns
    ----------
    probabilities: 1D np.array of float
        Probability between 0 (non-Ia) and 1 (Ia).

    Examples
    ----------
    >>> from fink_science.utilities import concat_col
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
    >>> df = df.withColumn('pIa', rfscore_sigmoid(*args))

    # Note that we can also specify a model
    >>> args = [F.col(i) for i in what_prefix] + [F.lit(model_path_sigmoid)]
    >>> df = df.withColumn('pIa', rfscore_sigmoid(*args))

    # Drop temp columns
    >>> df = df.drop(*what_prefix)

    >>> df.agg({"pIa": "min"}).collect()[0][0]
    0.0

    >>> df.agg({"pIa": "max"}).collect()[0][0] < 1.0
    True
    """
    # Flag empty alerts
    mask = magpsf.apply(lambda x: np.sum(np.array(x) == np.array(x))) > 3
    if len(jd[mask]) == 0:
        return pd.Series(np.zeros(len(jd), dtype=float))

    # Load pre-trained model `clf`
    if model is not None:
        clf = load_scikit_model(model.values[0])
    else:
        curdir = os.path.dirname(os.path.abspath(__file__))
        model = curdir + '/../data/models/default-model_sigmoid.obj'
        clf = load_scikit_model(model)

    test_features = []
    ids = pd.Series(range(len(jd)))
    for id in ids[mask]:
        # compute flux and flux error
        data = [mag2fluxcal_snana(*args) for args in zip(
            magpsf[id],
            sigmapsf[id])]
        flux, error = np.transpose(data)

        # make a Pandas DataFrame with exploded series
        pdf = pd.DataFrame.from_dict({
            'SNID': [id] * len(flux),
            'MJD': jd[id],
            'FLUXCAL': flux,
            'FLUXCALERR': error,
            'FLT': pd.Series(fid[id]).replace({1: 'g', 2: 'r'})
        })

        features = get_sigmoid_features_dev(pdf)
        test_features.append(features)

    # Make predictions
    probabilities = clf.predict_proba(test_features)

    # Take only probabilities to be Ia
    to_return = np.zeros(len(jd), dtype=float)
    to_return[mask] = probabilities.T[0]

    return pd.Series(to_return)


if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    ztf_alert_sample = 'fink_science/data/alerts/alerts.parquet'
    globs["ztf_alert_sample"] = ztf_alert_sample

    model_path_bazin = 'fink_science/data/models/default-model_bazin.obj'
    globs["model_path_bazin"] = model_path_bazin

    model_path_sigmoid = 'fink_science/data/models/default-model_sigmoid.obj'
    globs["model_path_sigmoid"] = model_path_sigmoid

    # Run the test suite
    spark_unit_tests(globs)
