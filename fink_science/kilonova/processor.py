# Copyright 2021-2022 AstroLab Software
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

from fink_utils.photometry.conversion import mag2fluxcal_snana
from fink_utils.xmatch.simbad import return_list_of_eg_host

from fink_science import __file__

from kndetect.utils import load_pcs
from kndetect.predict import load_classifier, predict_kn_score
from kndetect.features import extract_features_all_lightcurves, get_feature_names

from fink_science.tester import spark_unit_tests

@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def knscore(jd, fid, magpsf, sigmapsf, jdstarthist, cdsxmatch, ndethist, model_name=None) -> pd.Series:
    """ Return the probability of an alert to be a Kilonova using a Random
    Forest Classifier.

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
    jdstarthist: Spark DataFrame Column
        Column containing first time variability has been seen
    model_name: str
        Nome of the model to be fetched from the kndetect package.
        supported options: "complete.pkl", "partial.pkl"
        deault is "partial.pkl" (model trained for complete light curves)

    Returns
    ----------
    probabilities: 1D np.array of float
        Probability between 0 (non-KNe) and 1 (KNe).

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
    >>> args += [F.col('candidate.jdstarthist'), F.col('cdsxmatch'), F.col('candidate.ndethist')]
    >>> df = df.withColumn('pKNe', knscore(*args))

    >>> df.filter(df['pKNe'] > 0.5).count()
    1

    >>> df.filter(df['pKNe'] > 0.5).select(['rf_kn_vs_nonkn', 'pKNe']).show()
    +--------------+------------------+
    |rf_kn_vs_nonkn|              pKNe|
    +--------------+------------------+
    |           0.0|0.6333333333333333|
    +--------------+------------------+
    <BLANKLINE>

    # Perform the fit + classification (another model)
    >>> args = [F.col(i) for i in what_prefix]
    >>> args += [F.col('candidate.jdstarthist'), F.col('cdsxmatch'), F.col('candidate.ndethist')]
    >>> args += [F.lit('partial.pkl')]
    >>> df = df.withColumn('pKNe', knscore(*args))

    >>> df.filter(df['pKNe'] > 0.5).count()
    1
    """
    # Flag empty alerts
    mask = magpsf.apply(lambda x: np.sum(np.array(x) == np.array(x))) > 1

    mask *= (ndethist.astype(int) <= 20)

    mask *= jd.apply(lambda x: float(x[-1])) - jdstarthist.astype(float) < 20

    list_of_kn_host = return_list_of_eg_host()
    mask *= cdsxmatch.apply(lambda x: x in list_of_kn_host)

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

    # Load pre-trained model
    if model_name is None:
        model = load_classifier("partial.pkl")
    else:
        model = load_classifier(model_name.values[0])

    # Load pcs
    pcs = load_pcs()

    # define filters
    filters = ['g', 'r']

    # extract features (all filters) for each SNID
    features_df = extract_features_all_lightcurves(pdf, key="SNID", pcs=pcs, filters=filters)

    # make predictions
    # If coefficients in any band in zero, the event is predicted as non-KN by default
    probabilities_, _ = predict_kn_score(clf=model, features_df=features_df)

    # Take only probabilities to be KN
    to_return = np.zeros(len(jd), dtype=float)
    to_return[mask] = probabilities_.T[1]

    return pd.Series(to_return)

@pandas_udf(StringType(), PandasUDFType.SCALAR)
def extract_features_knscore(jd, fid, magpsf, sigmapsf) -> pd.Series:
    """ Extract features used by the Kilonova classifier (using a Random
    Forest Classifier).

    Parameters
    ----------
    jd: Spark DataFrame Column
        JD times (float)
    fid: Spark DataFrame Column
        Filter IDs (int)
    magpsf, sigmapsf: Spark DataFrame Columns
        Magnitude from PSF-fit photometry, and 1-sigma error

    Returns
    ----------
    out: str
        comma separated features

    Examples
    ----------
    >>> from pyspark.sql.functions import split
    >>> from pyspark.sql.types import FloatType
    >>> from fink_utils.spark.utils import concat_col
    >>> from kndetect.features import get_feature_names
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
    >>> df = df.withColumn('features', extract_features_knscore(*args))

    >>> KN_FEATURE_NAMES_3PC = get_feature_names(3)
    >>> for name in KN_FEATURE_NAMES_3PC:
    ...   index = KN_FEATURE_NAMES_3PC.index(name)
    ...   df = df.withColumn(name, split(df['features'], ',')[index].astype(FloatType()))

    # Trigger something
    >>> val = df.agg({KN_FEATURE_NAMES_3PC[0]: "min"}).collect()[0][0]
    >>> np.isclose(val, 0.0)
    True
    """
    # Flag empty alerts
    mask = magpsf.apply(lambda x: np.sum(np.array(x) == np.array(x))) > 1
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

    # Load pcs
    pcs = load_pcs()

    # define filters
    filters = ['g', 'r']

    # extract features (all filters) for each SNID
    features_df = extract_features_all_lightcurves(pdf, key="SNID", pcs=pcs, filters=filters)
    feature_col_names = get_feature_names()

    # return features for all events
    to_return_features = np.zeros(
        (len(jd), len(feature_col_names)),
        dtype=float
    )
    to_return_features[mask] = features_df[feature_col_names].values

    concatenated_features = [
        ','.join(np.array(i, dtype=str)) for i in to_return_features
    ]

    return pd.Series(concatenated_features)


if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    path = os.path.dirname(__file__)

    model_name = 'partial.pkl'
    globs["model_name"] = model_name

    ztf_alert_sample = 'file://{}/data/alerts/datatest'.format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
