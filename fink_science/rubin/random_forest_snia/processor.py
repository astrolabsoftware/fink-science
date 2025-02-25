# Copyright 2019-2025 AstroLab Software
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
from line_profiler import profile

import os
import pickle

from fink_science import __file__

from fink_utils.data.utils import format_data_as_snana
from fink_utils.data.utils import load_scikit_model

from actsnfink.rainbow import fit_rainbow

from actsnfink.classifier_sigmoid import RF_FEATURE_NAMES

from fink_science.tester import spark_unit_tests

RAINBOW_FEATURES_NAMES = [
    "amplitude",
    "rise_time",
    "Tmin",
    "delta_T",
    "k_sig",
    "reduced_chi2",
    "lc_max",
]


def extract_features_rainbow(
    midPointTai,
    filterName,
    cpsFlux,
    cpsFluxErr,
    band_wave_aa=None,
    with_baseline=None,
    min_data_points=None,
    low_bound=None,
) -> pd.Series:
    """Return the features used by the RF classifier for one alert.

    Notes
    -----
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
    -------
    features: list of floats
        Rainbow best-fit parameter values.

    Examples
    --------
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
    ...     a_feature = extract_features_rainbow(*[np.array(x) for x in alert.to_numpy()])
    ...     assert np.all(~np.isnan(a_feature))
    """
    if band_wave_aa is None:
        band_wave_aa = {
            "u": 3671.0,
            "g": 4827.0,
            "r": 6223.0,
            "i": 7546.0,
            "z": 8691.0,
            "Y": 9712.0,
        }
    if with_baseline is None:
        with_baseline = False
    if min_data_points is None:
        min_data_points = 7
    if low_bound is None:
        low_bound = -10

    if len(midPointTai) < min_data_points:
        return np.zeros(len(RAINBOW_FEATURES_NAMES), dtype=float)

    features = fit_rainbow(
        midPointTai,
        filterName,
        cpsFlux,
        cpsFluxErr,
        band_wave_aa=band_wave_aa,
        with_baseline=with_baseline,
        min_data_points=min_data_points,
        list_filters=band_wave_aa.keys(),
        low_bound=low_bound,
    )

    return features[1:]


@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
@profile
def rfscore_rainbow_elasticc(
    midPointTai,
    filterName,
    cpsFlux,
    cpsFluxErr,
    snr,
    hostgal_snsep,
    hostgal_zphot,
    maxduration=None,
    model=None,
    band_wave_aa=None,
    with_baseline=None,
    min_data_points=None,
    low_bound=None,
) -> pd.Series:
    """Return the probability of an alert to be a SNe Ia using a Random Forest Classifier (rainbow fit) on ELaSTICC alert data.

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
    -------
    probabilities: Spark DataFrame Column
        Probability between 0 (non-Ia) and 1 (Ia) for each alert.

    Examples
    --------
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

    >>> df.filter(df['pIa'] > 0.5).count()
    90

    >>> df.filter(df['pIa'] == -1.0).count()
    141
    """
    if band_wave_aa is None:
        band_wave_aa = pd.Series([
            {
                "u": 3671.0,
                "g": 4827.0,
                "r": 6223.0,
                "i": 7546.0,
                "z": 8691.0,
                "Y": 9712.0,
            }
        ])
    if with_baseline is None:
        with_baseline = pd.Series([False])
    if min_data_points is None:
        min_data_points = pd.Series([7])
    if low_bound is None:
        low_bound = pd.Series([-10])

    # dt is a column of floats
    dt = midPointTai.apply(lambda x: np.max(x) - np.min(x))

    # Maximum days in the history
    if maxduration is not None:
        mask = dt <= maxduration.to_numpy()[0]
    else:
        mask = np.repeat(True, len(midPointTai))

    if len(midPointTai[mask]) == 0:
        return pd.Series(np.zeros(len(midPointTai), dtype=float))

    # Load pre-trained model `clf`
    if model is not None:
        clf = load_scikit_model(model.to_numpy()[0])
    else:
        curdir = os.path.dirname(os.path.abspath(__file__))
        model = curdir + "/data/models/elasticc_rainbow_earlyIa_after_leak.pkl"
        clf = pickle.load(open(model, "rb"))

    candid = pd.Series(range(len(midPointTai)))
    ids = candid[mask]

    test_features = []
    flag = []
    for index in ids:
        features = extract_features_rainbow(
            midPointTai.to_numpy()[index],
            filterName.to_numpy()[index],
            cpsFlux.to_numpy()[index],
            cpsFluxErr.to_numpy()[index],
            band_wave_aa=band_wave_aa.to_numpy()[0],
            with_baseline=with_baseline.to_numpy()[0],
            min_data_points=min_data_points.to_numpy()[0],
            low_bound=low_bound.to_numpy()[0],
        )
        if features[0] == 0.0:
            flag.append(False)
        else:
            flag.append(True)

        meta_feats = [
            len(midPointTai.to_numpy()[index]),
            snr.to_numpy()[index],
            hostgal_snsep.to_numpy()[index],
            hostgal_zphot.to_numpy()[index],
        ]
        test_features.append(np.array(meta_feats + list(features)))

    flag = np.array(flag, dtype=bool)

    # Make predictions
    probabilities = clf.predict_proba(test_features)

    # pIa = -1.0 for objects that do not
    # have both features non-zero.
    probabilities[~flag] = [1.0, -1.0]

    # Take only probabilities to be Ia
    to_return = np.zeros(len(midPointTai), dtype=float)
    to_return[mask] = probabilities.T[1]

    return pd.Series(to_return)


@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
@profile
def rfscore_rainbow_elasticc_nometa(
    midPointTai,
    filterName,
    cpsFlux,
    cpsFluxErr,
    maxduration=None,
    model=None,
    band_wave_aa=None,
    with_baseline=None,
    min_data_points=None,
    low_bound=None,
) -> pd.Series:
    """Return the probability of an alert to be a SNe Ia using a Random Forest Classifier (rainbow fit) on ELaSTICC alert data.

    Parameters
    ----------
    midPointTai: Spark DataFrame Column
        JD times (vectors of floats)
    filterName: Spark DataFrame Column
        Filter IDs (vectors of str)
    cpsFlux, cpsFluxErr: Spark DataFrame Columns
        Magnitude from PSF-fit photometry, and 1-sigma error
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
    -------
    probabilities: Spark DataFrame Column
        Probability between 0 (non-Ia) and 1 (Ia) for each alert.

    Examples
    --------
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
    >>> df = df.withColumn('pIa', rfscore_rainbow_elasticc_nometa(*args))

    >>> df.filter(df['pIa'] > 0.6).count()
    36

    >>> df.filter(df['pIa'] == -1.0).count()
    141
    """
    if band_wave_aa is None:
        band_wave_aa = pd.Series([
            {
                "u": 3671.0,
                "g": 4827.0,
                "r": 6223.0,
                "i": 7546.0,
                "z": 8691.0,
                "Y": 9712.0,
            }
        ])
    if with_baseline is None:
        with_baseline = pd.Series([False])
    if min_data_points is None:
        min_data_points = pd.Series([7])
    if low_bound is None:
        low_bound = pd.Series([-10])

    # dt is a column of floats
    dt = midPointTai.apply(lambda x: np.max(x) - np.min(x))

    # Maximum days in the history
    if maxduration is not None:
        mask = dt <= maxduration.to_numpy()[0]
    else:
        mask = np.repeat(True, len(midPointTai))

    if len(midPointTai[mask]) == 0:
        return pd.Series(np.zeros(len(midPointTai), dtype=float))

    # Load pre-trained model `clf`
    if model is not None:
        clf = load_scikit_model(model.to_numpy()[0])
    else:
        curdir = os.path.dirname(os.path.abspath(__file__))
        model = curdir + "/data/models/elasticc_rainbow_earlyIa_nometa.pkl"
        clf = pickle.load(open(model, "rb"))

    candid = pd.Series(range(len(midPointTai)))
    ids = candid[mask]

    test_features = []
    flag = []
    for index in ids:
        features = extract_features_rainbow(
            midPointTai.to_numpy()[index],
            filterName.to_numpy()[index],
            cpsFlux.to_numpy()[index],
            cpsFluxErr.to_numpy()[index],
            band_wave_aa=band_wave_aa.to_numpy()[0],
            with_baseline=with_baseline.to_numpy()[0],
            min_data_points=min_data_points.to_numpy()[0],
            low_bound=low_bound.to_numpy()[0],
        )
        if features[0] == 0.0:
            flag.append(False)
        else:
            flag.append(True)

        meta_feats = [len(midPointTai.to_numpy()[index])]
        test_features.append(np.array(meta_feats + list(features)))

    flag = np.array(flag, dtype=bool)

    # Make predictions
    probabilities = clf.predict_proba(test_features)

    # pIa = -1.0 for objects that do not
    # have both features non-zero.
    probabilities[~flag] = [1.0, -1.0]

    # Take only probabilities to be Ia
    to_return = np.zeros(len(midPointTai), dtype=float)
    to_return[mask] = probabilities.T[1]

    return pd.Series(to_return)


if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = "file://{}/data/alerts/datatest".format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    elasticc_alert_sample = (
        "file://{}/data/alerts/test_elasticc_earlysnia.parquet".format(path)
    )
    globs["elasticc_alert_sample"] = elasticc_alert_sample

    model_path_sigmoid = "{}/data/models/default-model_sigmoid.obj".format(path)
    globs["model_path_sigmoid"] = model_path_sigmoid

    model_path_al_loop = "{}/data/models/for_al_loop/model_20240821.pkl".format(path)
    globs["model_path_al_loop"] = model_path_al_loop

    ztf_alert_with_i_band = (
        "file://{}/data/alerts/20240606_iband_history.parquet".format(path)
    )
    globs["ztf_alert_with_i_band"] = ztf_alert_with_i_band

    # Run the test suite
    spark_unit_tests(globs)
