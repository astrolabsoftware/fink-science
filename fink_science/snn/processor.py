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
from pyspark.sql.types import DoubleType, FloatType, ArrayType

from supernnova.validation.validate_onthefly import classify_lcs

import pandas as pd
import numpy as np

import os

from fink_science import __file__
from fink_science.snn.utilities import reformat_to_df

from fink_utils.data.utils import format_data_as_snana
from fink_utils.xmatch.simbad import return_list_of_eg_host

from fink_science.tester import spark_unit_tests

def apply_selection_cuts_ztf(
        magpsf: pd.Series, cdsxmatch: pd.Series,
        jd: pd.Series, jdstarthist: pd.Series, roid: pd.Series,
        minpoints: int = 2, maxndethist: int = 90) -> pd.Series:
    """ Apply selection cuts to keep only alerts of interest
    for SNN analysis

    Parameters
    ----------
    magpsf: pd.Series
        Series containing data measurement (array of double). Each row contains
        all measurement values for one alert.
    cdsxmatch: pd.Series
        Series containing crossmatch label with SIMBAD (str).
        Each row contains one label.
    jd: pd.Series
        Series containing JD values (array of float). Each row contains
        all measurement values for one alert.
    jdstarthist: pd.Series
        Series containing first JD for which the source varied (float).
        Each row contains one label.
    roid: pd.Series
        Series containing SSO label (int).
        Each row contains one label.

    Returns
    ---------
    mask: pd.Series
        Series containing `True` if the alert is valid, `False` otherwise.
        Each row contains one boolean.
    """
    # Flag empty alerts
    mask = magpsf.apply(lambda x: np.sum(np.array(x) == np.array(x))) >= minpoints

    mask *= jd.apply(lambda x: float(x[-1])) - jdstarthist.astype(float) <= maxndethist

    mask *= roid.astype(int) != 3

    list_of_sn_host = return_list_of_eg_host()
    mask *= cdsxmatch.apply(lambda x: x in list_of_sn_host)

    return mask

@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def snn_ia(candid, jd, fid, magpsf, sigmapsf, roid, cdsxmatch, jdstarthist, model_name, model_ext=None) -> pd.Series:
    """ Compute probabilities of alerts to be SN Ia using SuperNNova

    Parameters
    ----------
    candid: Spark DataFrame Column
        Candidate IDs (int64)
    jd: Spark DataFrame Column
        JD times (float)
    fid: Spark DataFrame Column
        Filter IDs (int)
    magpsf, sigmapsf: Spark DataFrame Columns
        Magnitude from PSF-fit photometry, and 1-sigma error
    model_name: Spark DataFrame Column
        SuperNNova pre-trained model. Currently available:
            * snn_snia_vs_nonia
            * snn_sn_vs_all
    model_ext: Spark DataFrame Column, optional
        Path to the trained model (overwrite `model`). Default is None

    Returns
    ----------
    probabilities: 1D np.array of float
        Probability between 0 (non-Ia) and 1 (Ia).

    Examples
    ----------
    >>> from fink_science.xmatch.processor import xmatch_cds
    >>> from fink_science.asteroids.processor import roid_catcher
    >>> from fink_utils.spark.utils import concat_col
    >>> from pyspark.sql import functions as F

    >>> df = spark.read.load(ztf_alert_sample)

    # Add SIMBAD field
    >>> df = xmatch_cds(df)

    # Required alert columns
    >>> what = ['jd', 'fid', 'magpsf', 'sigmapsf']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    # Add SSO field
    >>> args_roid = [
    ...    'cjd', 'cmagpsf',
    ...    'candidate.ndethist', 'candidate.sgscore1',
    ...    'candidate.ssdistnr', 'candidate.distpsnr1']
    >>> df = df.withColumn('roid', roid_catcher(*args_roid))

    # Perform the fit + classification (default model)
    >>> args = ['candid', 'cjd', 'cfid', 'cmagpsf', 'csigmapsf']
    >>> args += [F.col('roid'), F.col('cdsxmatch'), F.col('candidate.jdstarthist')]
    >>> args += [F.lit('snn_snia_vs_nonia')]
    >>> df = df.withColumn('pIa', snn_ia(*args))

    >>> df.filter(df['pIa'] > 0.5).count()
    6

    # Note that we can also specify a model
    >>> args = [F.col(i) for i in ['candid', 'cjd', 'cfid', 'cmagpsf', 'csigmapsf']]
    >>> args += [F.col('roid'), F.col('cdsxmatch'), F.col('candidate.jdstarthist')]
    >>> args += [F.lit(''), F.lit(model_path)]
    >>> df = df.withColumn('pIa2', snn_ia(*args))

    >>> df.filter(df['pIa2'] > 0.5).count()
    8
    """
    mask = apply_selection_cuts_ztf(magpsf, cdsxmatch, jd, jdstarthist, roid)

    if len(jd[mask]) == 0:
        return pd.Series(np.zeros(len(jd), dtype=float))

    candid = candid.apply(lambda x: str(x))
    pdf = format_data_as_snana(jd, magpsf, sigmapsf, fid, candid, mask)

    if model_ext is not None:
        # take the first element of the Series
        model = model_ext.values[0]
    else:
        # Load pre-trained model
        curdir = os.path.dirname(os.path.abspath(__file__))
        model = curdir + '/data/models/snn_models/{}/model.pt'.format(model_name.values[0])

    # Compute predictions
    pdf = pdf.dropna()
    ids, pred_probs = classify_lcs(pdf, model, 'cpu')

    # Reformat and re-index
    preds_df = reformat_to_df(pred_probs, ids=ids)
    preds_df.index = preds_df.SNID

    # Take only probabilities to be Ia
    to_return = np.zeros(len(jd), dtype=float)
    ia = preds_df.reindex([str(i) for i in candid[mask].values])
    to_return[mask] = ia.prob_class0.values

    # return probabilities to be Ia
    return pd.Series(to_return)

@pandas_udf(FloatType(), PandasUDFType.SCALAR)
def snn_ia_elasticc(
        diaSourceId, midPointTai, filterName, psFlux, psFluxErr,
        roid, cdsxmatch, jdstarthist,
        mwebv, redshift, redshift_err,
        model_name, model_ext=None) -> pd.Series:
    """ Compute probabilities of alerts to be SN Ia using SuperNNova

    Single-class model (default stored at `data/models/snn_models/elasticc_ia`)

    Parameters
    ----------
    diaSourceId: Spark DataFrame Column
        Candidate IDs (int64)
    midPointTai: Spark DataFrame Column
        JD times (float)
    filterName: Spark DataFrame Column
        Filter IDs (str)
    psFlux, psFluxErr: Spark DataFrame Columns
        SNANA calibrated flux from LSST, and 1-sigma error
    model_name: Spark DataFrame Column
        SuperNNova pre-trained model. Currently available:
            * elasticc
    model_ext: Spark DataFrame Column, optional
        Path to the trained model (overwrite `model_name`). Default is None

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
    >>> df = df.withColumn('roid', F.lit(0))

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

    # add redshift
    >>> df = df.withColumn('redshift', F.when(df['diaObject.hostgal_zspec'] != -9.0, df['diaObject.hostgal_zspec']).otherwise(df['diaObject.hostgal_zphot']))
    >>> df = df.withColumn('redshift_err', F.when(df['diaObject.hostgal_zspec_err'] != -9.0, df['diaObject.hostgal_zspec_err']).otherwise(df['diaObject.hostgal_zphot_err']))

    # Perform the fit + classification (default model)
    >>> args = [F.col('diaSource.diaSourceId')]
    >>> args += [F.col(i) for i in what_prefix]
    >>> args += [F.col('roid'), F.col('cdsxmatch'), F.array_min('cmidPointTai')]
    >>> args += [F.col('diaObject.mwebv'), F.col('redshift'), F.col('redshift_err')]
    >>> args += [F.lit('elasticc_ia')]
    >>> df = df.withColumn('pIa', snn_ia_elasticc(*args))

    >>> df.filter(df['pIa'] > 0.5).count()
    5
    """
    mask = apply_selection_cuts_ztf(
        psFlux, cdsxmatch, midPointTai, jdstarthist, roid, maxndethist=180)

    if len(midPointTai[mask]) == 0:
        return pd.Series(np.zeros(len(midPointTai), dtype=float))

    # Conversion to FLUXCAL
    fac = 10**(-(31.4 - 27.5) / 2.5)
    psFlux = psFlux * fac
    psFluxErr = psFluxErr * fac

    diaSourceId = diaSourceId.apply(lambda x: str(x))
    pdf = format_data_as_snana(
        midPointTai, psFlux, psFluxErr,
        filterName, diaSourceId, mask,
        transform_to_flux=False
    )

    # Add extinction & redshift
    pdf_tmp = pd.DataFrame.from_dict(
        {
            'jd': midPointTai[mask],
            'MWEBV': mwebv[mask],
            'HOSTGAL_SPECZ': redshift[mask],
            'HOSTGAL_PHOTOZ': redshift[mask],
            'HOSTGAL_SPECZ_ERR': redshift_err[mask],
            'HOSTGAL_PHOTOZ_ERR': redshift_err[mask],
        }
    )
    pdf_tmp = pdf_tmp.explode('jd')

    pdf['MWEBV'] = pdf_tmp['MWEBV']
    pdf['HOSTGAL_SPECZ'] = pdf_tmp['HOSTGAL_SPECZ']
    pdf['HOSTGAL_SPECZ_ERR'] = pdf_tmp['HOSTGAL_SPECZ_ERR']
    pdf['HOSTGAL_PHOTOZ'] = pdf_tmp['HOSTGAL_PHOTOZ']
    pdf['HOSTGAL_PHOTOZ_ERR'] = pdf_tmp['HOSTGAL_PHOTOZ_ERR']

    if model_ext is not None:
        # take the first element of the Series
        model = model_ext.values[0]
    else:
        # Load pre-trained model
        curdir = os.path.dirname(os.path.abspath(__file__))
        model = curdir + '/data/models/snn_models/{}/model.pt'.format(model_name.values[0])

    # Compute predictions
    if len(pdf) == 0:
        return pd.Series(np.zeros(len(midPointTai), dtype=float))

    ids, pred_probs = classify_lcs(pdf, model, 'cpu')

    # Reformat and re-index
    preds_df = reformat_to_df(pred_probs, ids=ids)
    preds_df.index = preds_df.SNID

    # Take only probabilities to be Ia
    to_return = np.zeros(len(midPointTai), dtype=float)
    ia = preds_df.reindex([str(i) for i in diaSourceId[mask].values])
    to_return[mask] = ia.prob_class0.values

    # return probabilities to be Ia
    return pd.Series(to_return)

def extract_max_prob(arr):
    """ Extract main class and associated probability from a vector of probabilities
    """
    array = np.array(arr)
    index = np.argmax(array)
    return {'class': index, 'prob': array[index]}

@pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR)
def snn_broad_elasticc(
        diaSourceId, midPointTai, filterName, psFlux, psFluxErr,
        roid, cdsxmatch, jdstarthist,
        mwebv, redshift, redshift_err,
        model_name, model_ext=None) -> pd.Series:
    """ Compute main class and associated probability for each alert

    Multi-class model (default stored at `data/models/snn_models/elasticc_broad`)

    Parameters
    ----------
    diaSourceId: Spark DataFrame Column
        Candidate IDs (int64)
    midPointTai: Spark DataFrame Column
        JD times (float)
    filterName: Spark DataFrame Column
        Filter IDs (str)
    psFlux, psFluxErr: Spark DataFrame Columns
        SNANA calibrated flux from LSST, and 1-sigma error
    model_name: Spark DataFrame Column
        SuperNNova pre-trained model. Currently available:
            * elasticc
    model_ext: Spark DataFrame Column, optional
        Path to the trained model (overwrite `model_name`). Default is None

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
    >>> df = df.withColumn('roid', F.lit(0))

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

    # add redshift
    >>> df = df.withColumn('redshift', F.when(df['diaObject.hostgal_zspec'] != -9.0, df['diaObject.hostgal_zspec']).otherwise(df['diaObject.hostgal_zphot']))
    >>> df = df.withColumn('redshift_err', F.when(df['diaObject.hostgal_zspec_err'] != -9.0, df['diaObject.hostgal_zspec_err']).otherwise(df['diaObject.hostgal_zphot_err']))

    # Perform the fit + classification (default model)
    >>> args = [F.col('diaSource.diaSourceId')]
    >>> args += [F.col(i) for i in what_prefix]
    >>> args += [F.col('roid'), F.col('cdsxmatch'), F.array_min('cmidPointTai')]
    >>> args += [F.col('diaObject.mwebv'), F.col('redshift'), F.col('redshift_err')]
    >>> args += [F.lit('elasticc_broad')]
    >>> df = df.withColumn('preds', snn_broad_elasticc(*args))

    >>> df = df.withColumn('snn_class', F.col('preds').getItem(0).astype('int'))
    >>> df = df.withColumn('snn_max_prob', F.col('preds').getItem(1))

    >>> df.filter(df['snn_class'] == 0).count()
    19
    """
    mask = apply_selection_cuts_ztf(
        psFlux, cdsxmatch, midPointTai, jdstarthist, roid, maxndethist=1e6)

    if len(midPointTai[mask]) == 0:
        snn_class = np.ones(len(midPointTai), dtype=float) * -1
        snn_max_prob = np.zeros(len(midPointTai), dtype=float)
        return pd.Series([[i, j] for i, j in zip(snn_class, snn_max_prob)])

    # Conversion to FLUXCAL
    fac = 10**(-(31.4 - 27.5) / 2.5)
    psFlux = psFlux * fac
    psFluxErr = psFluxErr * fac

    diaSourceId = diaSourceId.apply(lambda x: str(x))
    pdf = format_data_as_snana(
        midPointTai, psFlux, psFluxErr,
        filterName, diaSourceId, mask,
        transform_to_flux=False
    )

    # Add extinction & redshift
    pdf_tmp = pd.DataFrame.from_dict(
        {
            'jd': midPointTai[mask],
            'MWEBV': mwebv[mask],
            'HOSTGAL_SPECZ': redshift[mask],
            'HOSTGAL_PHOTOZ': redshift[mask],
            'HOSTGAL_SPECZ_ERR': redshift_err[mask],
            'HOSTGAL_PHOTOZ_ERR': redshift_err[mask],
        }
    )
    pdf_tmp = pdf_tmp.explode('jd')

    pdf['MWEBV'] = pdf_tmp['MWEBV']
    pdf['HOSTGAL_SPECZ'] = pdf_tmp['HOSTGAL_SPECZ']
    pdf['HOSTGAL_SPECZ_ERR'] = pdf_tmp['HOSTGAL_SPECZ_ERR']
    pdf['HOSTGAL_PHOTOZ'] = pdf_tmp['HOSTGAL_PHOTOZ']
    pdf['HOSTGAL_PHOTOZ_ERR'] = pdf_tmp['HOSTGAL_PHOTOZ_ERR']

    if model_ext is not None:
        # take the first element of the Series
        model = model_ext.values[0]
    else:
        # Load pre-trained model
        curdir = os.path.dirname(os.path.abspath(__file__))
        model = curdir + '/data/models/snn_models/{}/model.pt'.format(model_name.values[0])

    # Compute predictions
    if len(pdf) == 0:
        snn_class = np.ones(len(midPointTai), dtype=float) * -1
        snn_max_prob = np.zeros(len(midPointTai), dtype=float)
        return pd.Series([[i, j] for i, j in zip(snn_class, snn_max_prob)])

    ids, pred_probs = classify_lcs(pdf, model, 'cpu')

    # Reformat and re-index
    preds_df = reformat_to_df(pred_probs, ids=ids)
    preds_df.index = preds_df.SNID

    # Take only probabilities to be Ia
    snn_class = np.ones(len(midPointTai), dtype=float) * -1
    snn_max_prob = np.zeros(len(midPointTai), dtype=float)

    all_preds = preds_df.reindex([str(i) for i in diaSourceId[mask].values])

    cols = ['prob_class{}'.format(i) for i in range(5)]
    all_preds[['snn_class', 'snn_max_prob']] = all_preds[cols].apply(lambda x: extract_max_prob(x), axis=1, result_type="expand")
    snn_class[mask] = all_preds.snn_class.values
    snn_max_prob[mask] = all_preds.snn_max_prob.values

    # return main class and associated probability
    return pd.Series([[i, j] for i, j in zip(snn_class, snn_max_prob)])


if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = 'file://{}/data/alerts/datatest'.format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    elasticc_alert_sample = 'file://{}/data/alerts/elasticc_sample_seed0.parquet'.format(path)
    globs["elasticc_alert_sample"] = elasticc_alert_sample

    model_path = '{}/data/models/snn_models/snn_sn_vs_all/model.pt'.format(path)
    globs["model_path"] = model_path

    elasticc_model_path = '{}/data/models/snn_models/elasticc/model.pt'.format(path)
    globs["elasticc_model_path"] = elasticc_model_path

    # Run the test suite
    spark_unit_tests(globs)
