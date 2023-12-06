# Copyright 2019-2023 AstroLab Software
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
import pickle

from fink_science import __file__

from fink_utils.data.utils import format_data_as_snana
from fink_utils.data.utils import load_scikit_model
from fink_utils.xmatch.simbad import return_list_of_eg_host

from actsnfink.classifier_sigmoid import get_sigmoid_features_dev
# from actsnfink.classifier_sigmoid import get_sigmoid_features_elasticc_perfilter
from actsnfink.rainbow import fit_rainbow

from actsnfink.classifier_sigmoid import RF_FEATURE_NAMES

from fink_science.tester import spark_unit_tests

RAINBOW_FEATURES_NAMES = [
    "amplitude", "rise_time",
    "Tmin", "delta_T", "k_sig",
    "reduced_chi2", "lc_max"
]


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
def rfscore_sigmoid_full(
        jd, fid, magpsf, sigmapsf, cdsxmatch, ndethist,
        min_rising_points=pd.Series([2]),
        min_data_points=pd.Series([4]),
        rising_criteria=pd.Series(['ewma']),
        model=None) -> pd.Series:
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
    min_rising_points, min_data_points: int
        Parameters from fink_sn_activelearning.git
    rising_criteria: str
        How to compute derivatives: ewma (default), or diff.
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

    # We can also specify fink_sn_activelearning parameters
    >>> args = [F.col(i) for i in what_prefix]
    >>> args += [F.col('cdsxmatch'), F.col('candidate.ndethist')]
    >>> args += [F.lit(2), F.lit(4), F.lit('ewma')]
    >>> df = df.withColumn('pIa', rfscore_sigmoid_full(*args))

    >>> df.filter(df['pIa'] > 0.5).count()
    6

    # We can also specify a different model
    >>> args = [F.col(i) for i in what_prefix]
    >>> args += [F.col('cdsxmatch'), F.col('candidate.ndethist')]
    >>> args += [F.lit(1), F.lit(3), F.lit('diff')]
    >>> args += [F.lit(model_path_al_loop)]
    >>> df = df.withColumn('pIaAL', rfscore_sigmoid_full(*args))

    >>> df.filter(df['pIaAL'] > 0.5).count()
    5

    >>> df.agg({"pIaAL": "max"}).collect()[0][0] < 1.0
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
        features = get_sigmoid_features_dev(
            pdf_sub,
            min_rising_points=min_rising_points.values[0],
            min_data_points=min_data_points.values[0],
            rising_criteria=rising_criteria.values[0]
        )
        if (features[0] == 0) or (features[6] == 0):
            flag.append(False)
        else:
            flag.append(True)
        test_features.append(features)

    flag = np.array(flag, dtype=bool)

    # Make predictions
    probabilities = clf.predict_proba(test_features)

    # pIa = 0.0 for objects that do not
    # have both features non-zero.
    probabilities[~flag] = [1.0, 0.0]

    # Take only probabilities to be Ia
    to_return = np.zeros(len(jd), dtype=float)
    to_return[mask] = probabilities.T[1]

    return pd.Series(to_return)

@pandas_udf(StringType(), PandasUDFType.SCALAR)
def extract_features_rf_snia(
        jd, fid, magpsf, sigmapsf, cdsxmatch, ndethist,
        min_rising_points=pd.Series([2]),
        min_data_points=pd.Series([4]),
        rising_criteria=pd.Series(['ewma'])) -> pd.Series:
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
    min_rising_points, min_data_points: int
        Parameters from fink_sn_activelearning.git
    rising_criteria: str
        How to compute derivatives: ewma (default), or diff.

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
        features = get_sigmoid_features_dev(
            pdf_sub,
            min_rising_points=min_rising_points.values[0],
            min_data_points=min_data_points.values[0],
            rising_criteria=rising_criteria.values[0]
        )
        test_features.append(features)

    to_return_features = np.zeros((len(jd), len(RF_FEATURE_NAMES)), dtype=float)
    to_return_features[mask] = test_features

    concatenated_features = [
        ','.join(np.array(i, dtype=str)) for i in to_return_features
    ]

    return pd.Series(concatenated_features)

def extract_features_rainbow(
        midPointTai, filterName, cpsFlux, cpsFluxErr,
        band_wave_aa={'u': 3671.0, 'g': 4827.0, 'r': 6223.0, 'i': 7546.0, 'z': 8691.0, 'Y': 9712.0},
        with_baseline=False,
        min_data_points=7,
        low_bound=-10) -> pd.Series:
    """ Return the features used by the RF classifier for one alert.

    Features (incl. order) are given by `RAINBOW_FEATURES_NAMES`.

    Parameters
    ----------
    midPointTai: np.array of floats
        MJD vector for one object
    filterName: np.array of str
        Filter name vector for one object
    cpsFlux, cpsFluxErr: np.array of float
        Flux from PSF-fit photometry, and 1-sigma error
    band_wave_aa: dict (optional)
        Dictionary with effective wavelength for each filter.
        Default is for ZTF: {"g": 4770.0, "r": 6231.0, "i": 7625.0}
    with_baseline: bool (optional)
        Baseline to be considered. Default is False (baseline 0).
    min_data_points: int (optional)
       Minimum number of data points in all filters. Default is 7.
    low_bound: float (optional)
        Lower bound of FLUXCAL to consider. Default is -10.

    Returns
    ----------
    features: list of floats
        Rainbow best-fit parameter values.

    Examples
    ----------
    >>> from pyspark.sql.functions import split
    >>> from pyspark.sql.types import FloatType
    >>> from fink_utils.spark.utils import concat_col
    >>> from pyspark.sql import functions as F

    >>> df = spark.read.load(elasticc_alert_sample)

    # Required alert columns
    >>> what = ['midPointTai', 'filterName', 'psFlux', 'psFluxErr']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...     df = concat_col(
    ...         df, colname, prefix=prefix,
    ...         current='diaSource', history='prvDiaForcedSources')

    >>> pdf = df.select(what_prefix).toPandas()

    # Test no NaNs
    >>> for index, alert in pdf.iterrows():
    ...     a_feature = extract_features_rainbow(*[np.array(x) for x in alert.values])
    ...     assert np.all(~np.isnan(a_feature))
    """
    if len(midPointTai) < min_data_points:
        return np.zeros(len(RAINBOW_FEATURES_NAMES), dtype=float)

    features = fit_rainbow(
        midPointTai, filterName, cpsFlux, cpsFluxErr,
        band_wave_aa=band_wave_aa,
        with_baseline=with_baseline,
        min_data_points=min_data_points,
        list_filters=band_wave_aa.keys(),
        low_bound=low_bound
    )

    return features[1:]


@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def rfscore_rainbow_elasticc(
        midPointTai, filterName, cpsFlux, cpsFluxErr,
        snr,
        hostgal_snsep,
        hostgal_zphot,
        maxduration=None,
        model=None,
        band_wave_aa=pd.Series([{'u': 3671.0, 'g': 4827.0, 'r': 6223.0, 'i': 7546.0, 'z': 8691.0, 'Y': 9712.0}]),
        with_baseline=pd.Series([False]),
        min_data_points=pd.Series([7]),
        low_bound=pd.Series([-10])) -> pd.Series:
    """ Return the probability of an alert to be a SNe Ia using a Random
    Forest Classifier (rainbow fit) on ELaSTICC alert data.

    Parameters
    ----------
    midPointTai: Spark DataFrame Column
        JD times (vectors of floats)
    filterName: Spark DataFrame Column
        Filter IDs (vectors of str)
    cpsFlux, cpsFluxErr: Spark DataFrame Columns
        Magnitude from PSF-fit photometry, and 1-sigma error
    snr: Spark DataFrame Column
        SNR from `diaSource` (float)
    hostgal_snsep: Spark DataFrame Column
        `hostgal_snsep` from `diaObject` (float)
    hostgal_zphot: Spark DataFrame Column
        `hostgal_zphot` from `diaObject` (float)
    maxduration: Spark DataFrame Column
        Maximum duration in days to consider the object for classification (int).
        Default is None, meaning no maximum duration applied.
    model: Spark DataFrame Column, optional
        Path to the trained model. Default is None, in which case the default
        model `data/models/default-model.obj` is loaded.
    band_wave_aa: Spark DataFrame Column
        Dictionary with effective wavelength for each filter.
        Default is for Elasticc.
    with_baseline: Spark DataFrame Column
        Baseline to be considered (bool). Default is False (baseline 0 in flux space).
    min_data_points: Spark DataFrame Column
       Minimum number of data points in all filters. Default is 7.
    low_bound: Spark DataFrame Column
        Lower bound of FLUXCAL to consider (float). Default is -10.

    Returns
    ----------
    probabilities: Spark DataFrame Column
        Probability between 0 (non-Ia) and 1 (Ia) for each alert.

    Examples
    ----------
    >>> from fink_utils.spark.utils import concat_col
    >>> from pyspark.sql import functions as F

    >>> df = spark.read.format('parquet').load(elasticc_alert_sample)

    # Required alert columns
    >>> what = ['midPointTai', 'filterName', 'psFlux', 'psFluxErr']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...     df = concat_col(
    ...         df, colname, prefix=prefix,
    ...         current='diaSource', history='prvDiaForcedSources')

    # Perform the fit + classification (default model)
    >>> args = [F.col(i) for i in what_prefix]
    >>> args += [F.col('diaSource.snr')]
    >>> args += [F.col('diaObject.hostgal_snsep')]
    >>> args += [F.col('diaObject.hostgal_zphot')]
    >>> df = df.withColumn('pIa', rfscore_rainbow_elasticc(*args))
    >>> df.select('pIa').show()
    >>> df.filter(df['pIa'] > 0.5).count()
    80
    """
    # dt is a column of floats
    dt = midPointTai.apply(lambda x: np.max(x) - np.min(x))

    # Maximum days in the history
    if maxduration is not None:
        mask = (dt <= maxduration.values[0])
    else:
        mask = np.repeat(True, len(midPointTai))

    if len(midPointTai[mask]) == 0:
        return pd.Series(np.zeros(len(midPointTai), dtype=float))

    # Load pre-trained model `clf`
    if model is not None:
        clf = load_scikit_model(model.values[0])
    else:
        curdir = os.path.dirname(os.path.abspath(__file__))
        model = curdir + '/data/models/elasticc_rainbow_earlyIa.pkl'
        clf = pickle.load(open(model, 'rb'))

    candid = pd.Series(range(len(midPointTai)))
    ids = candid[mask]

    test_features = []
    for index in ids:
        features = extract_features_rainbow(
            midPointTai.values[index],
            filterName.values[index],
            cpsFlux.values[index],
            cpsFluxErr.values[index],
            band_wave_aa=band_wave_aa.values[0],
            with_baseline=with_baseline.values[0],
            min_data_points=min_data_points.values[0],
            low_bound=low_bound.values[0]
        )

        meta_feats = [
            len(midPointTai.values[index]),
            snr.values[index],
            hostgal_snsep.values[index],
            hostgal_zphot.values[index]
        ]
        # test_features.append(meta_feats + list(features[1:]))
        test_features.append(np.array(meta_feats + list(features)))

    # Make predictions
    probabilities = clf.predict_proba(test_features)

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

    elasticc_alert_sample = 'file://{}/data/alerts/test_elasticc_earlysnia.parquet'.format(path)
    globs["elasticc_alert_sample"] = elasticc_alert_sample

    model_path_sigmoid = '{}/data/models/default-model_sigmoid.obj'.format(path)
    globs["model_path_sigmoid"] = model_path_sigmoid

    model_path_al_loop = '{}/data/models/for_al_loop/model_20231009.pkl'.format(path)
    globs["model_path_al_loop"] = model_path_al_loop

    # Run the test suite
    spark_unit_tests(globs)
