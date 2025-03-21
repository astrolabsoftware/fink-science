# Copyright 2020-2025 AstroLab Software
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
from line_profiler import profile

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import FloatType, ArrayType

from supernnova.validation.validate_onthefly import classify_lcs

import pandas as pd
import numpy as np

import os

from fink_science import __file__
from fink_science.rubin.snn.utilities import reformat_to_df

from fink_utils.data.utils import format_data_as_snana

from fink_science.tester import spark_unit_tests


@pandas_udf(FloatType(), PandasUDFType.SCALAR)
@profile
def snn_ia_elasticc(
    diaSourceId,
    midpointMjdTai,
    band,
    psfFlux,
    psfFluxErr,
    model_name,
    model_ext=None,
) -> pd.Series:
    """Compute probabilities of alerts to be SN Ia using SuperNNova

    Single-class model (default stored at `data/models/snn_models/elasticc_ia`)

    Parameters
    ----------
    diaSourceId: Spark DataFrame Column
        Candidate IDs (int64)
    midpointMjdTai: Spark DataFrame Column
        JD times (float)
    band: Spark DataFrame Column
        Filter IDs (str)
    psfFlux, psfFluxErr: Spark DataFrame Columns
        SNANA calibrated flux from LSST, and 1-sigma error
    model_name: Spark DataFrame Column
        SuperNNova pre-trained model. Currently available:
            * elasticc
    model_ext: Spark DataFrame Column, optional
        Path to the trained model (overwrite `model_name`). Default is None

    Returns
    -------
    probabilities: 1D np.array of float
        Probability between 0 (non-Ia) and 1 (Ia).

    Examples
    --------
    >>> from fink_utils.spark.utils import concat_col
    >>> from pyspark.sql import functions as F

    >>> df = spark.read.format('parquet').load(rubin_alert_sample)

    # Required alert columns
    >>> what = ['midpointMjdTai', 'band', 'psfFlux', 'psfFluxErr']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...     df = concat_col(
    ...         df, colname, prefix=prefix,
    ...         current='diaSource', history='prvDiaForcedSources')

    # Does not work for the moment
    # # Perform the fit + classification: Ia vs all
    # >>> args = [F.col('diaSource.diaSourceId')]
    # >>> args += [F.col(i) for i in what_prefix]
    # >>> args += [F.lit('elasticc_ia')]
    # >>> df = df.withColumn('ia_vs_all', snn_ia_elasticc(*args))

    # >>> df.filter(df['ia_vs_all'] > 0.5).count()
    # 1

    # Perform the fit + classification: SN vs all
    >>> args = [F.col('diaSource.diaSourceId')]
    >>> args += [F.col(i) for i in what_prefix]
    >>> args += [F.lit('elasticc_binary_broad/SN_vs_other')]
    >>> df = df.withColumn('sn_vs_all', snn_ia_elasticc(*args))

    >>> df.filter(df['sn_vs_all'] > 0.5).count()
    48
    """
    # No a priori cuts
    mask = np.ones(len(diaSourceId), dtype=bool)

    if len(midpointMjdTai[mask]) == 0:
        return pd.Series(np.zeros(len(midpointMjdTai), dtype=float))

    # Conversion to FLUXCAL
    fac = 10 ** (-(31.4 - 27.5) / 2.5)
    psfFlux = psfFlux * fac
    psfFluxErr = psfFluxErr * fac

    diaSourceId = diaSourceId.apply(lambda x: str(x))
    pdf = format_data_as_snana(
        midpointMjdTai,
        psfFlux,
        psfFluxErr,
        band,
        diaSourceId,
        mask,
        transform_to_flux=False,
    )

    if model_ext is not None:
        # take the first element of the Series
        model = model_ext.to_numpy()[0]
    else:
        # Load pre-trained model
        curdir = os.path.dirname(os.path.abspath(__file__))
        model = curdir + "/data/models/snn_models/{}/model.pt".format(
            model_name.to_numpy()[0]
        )

    # Compute predictions
    if len(pdf) == 0:
        return pd.Series(np.zeros(len(midpointMjdTai), dtype=float))

    ids, pred_probs = classify_lcs(pdf, model, "cpu")

    # Reformat and re-index
    preds_df = reformat_to_df(pred_probs, ids=ids)
    preds_df.index = preds_df.SNID

    # Take only probabilities to be Ia
    to_return = np.zeros(len(midpointMjdTai), dtype=float)
    ia = preds_df.reindex([str(i) for i in diaSourceId[mask].to_numpy()])
    to_return[mask] = ia.prob_class0.to_numpy()

    # return probabilities to be Ia
    return pd.Series(to_return)


def extract_max_prob(arr):
    """Extract main class and associated probability from a vector of probabilities"""
    array = np.array(arr)
    index = np.argmax(array)
    return {"class": index, "prob": array[index]}


@pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR)
@profile
def snn_broad_elasticc(
    diaSourceId,
    midpointMjdTai,
    band,
    psfFlux,
    psfFluxErr,
    model_name,
    model_ext=None,
) -> pd.Series:
    """Compute main class and associated probability for each alert

    Multi-class model (default stored at `data/models/snn_models/elasticc_broad`)

    Parameters
    ----------
    diaSourceId: Spark DataFrame Column
        Candidate IDs (int64)
    midpointMjdTai: Spark DataFrame Column
        JD times (float)
    band: Spark DataFrame Column
        Filter IDs (str)
    psfFlux, psfFluxErr: Spark DataFrame Columns
        SNANA calibrated flux from LSST, and 1-sigma error
    model_name: Spark DataFrame Column
        SuperNNova pre-trained model. Currently available:
            * elasticc
    model_ext: Spark DataFrame Column, optional
        Path to the trained model (overwrite `model_name`). Default is None

    Returns
    -------
    probabilities: 1D np.array of float
        Probability between 0 (non-Ia) and 1 (Ia).

    """
    # No a priori cuts
    mask = np.ones(len(diaSourceId), dtype=bool)

    if len(midpointMjdTai[mask]) == 0:
        return pd.Series([[0.0 for i in range(5)] for j in range(len(diaSourceId))])

    # Conversion to FLUXCAL
    fac = 10 ** (-(31.4 - 27.5) / 2.5)
    psfFlux = psfFlux * fac
    psfFluxErr = psfFluxErr * fac

    diaSourceId = diaSourceId.apply(lambda x: str(x))
    pdf = format_data_as_snana(
        midpointMjdTai,
        psfFlux,
        psfFluxErr,
        band,
        diaSourceId,
        mask,
        transform_to_flux=False,
    )

    if model_ext is not None:
        # take the first element of the Series
        model = model_ext.to_numpy()[0]
    else:
        # Load pre-trained model
        curdir = os.path.dirname(os.path.abspath(__file__))
        model = curdir + "/data/models/snn_models/{}/model.pt".format(
            model_name.to_numpy()[0]
        )

    # Compute predictions
    if len(pdf) == 0:
        snn_class = np.ones(len(midpointMjdTai), dtype=float) * -1
        snn_max_prob = np.zeros(len(midpointMjdTai), dtype=float)
        return pd.Series([[i, j] for i, j in zip(snn_class, snn_max_prob)])

    ids, pred_probs = classify_lcs(pdf, model, "cpu")

    # Reformat and re-index
    preds_df = reformat_to_df(pred_probs, ids=ids)
    preds_df.index = preds_df.SNID

    all_preds = preds_df.reindex([str(i) for i in diaSourceId[mask].to_numpy()])

    cols = ["prob_class{}".format(i) for i in range(5)]
    all_preds["all"] = all_preds[cols].to_numpy().tolist()

    return all_preds["all"]


if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    path = os.path.dirname(__file__)

    rubin_alert_sample = "file://{}/data/alerts/or4_lsst7.1".format(path)
    globs["rubin_alert_sample"] = rubin_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
