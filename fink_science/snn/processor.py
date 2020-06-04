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
from pyspark.sql.types import DoubleType

from supernnova.validation.validate_onthefly import classify_lcs

import pandas as pd
import numpy as np

import os

from fink_science.conversion import mag2fluxcal_snana
from fink_science.snn.utilities import reformat_to_df

from fink_science.tester import spark_unit_tests

@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def snn_ia(candid, jd, fid, magpsf, sigmapsf, model=None) -> pd.Series:
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
    model: Spark DataFrame Column, optional
        Path to the trained model. Default is None, in which case the default
        model `data/models/<vanilla_S_0_...>.pt` is loaded.

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
    >>> args = [F.col(i) for i in ['candid', 'cjd', 'cfid', 'cmagpsf', 'csigmapsf']]
    >>> df = df.withColumn('pIa', snn_ia(*args))

    # Note that we can also specify a model
    >>> args = [F.col(i) for i in ['candid', 'cjd', 'cfid', 'cmagpsf', 'csigmapsf']] + [F.lit(model_path)]
    >>> df = df.withColumn('pIa', snn_ia(*args))

    # Drop temp columns
    >>> df = df.drop(*what_prefix)

    >>> df.agg({"pIa": "min"}).collect()[0][0] >= 0.0
    True

    >>> df.agg({"pIa": "max"}).collect()[0][0] < 1.0
    True
    """
    # Load pre-trained model
    if model is None:
        curdir = os.path.dirname(os.path.abspath(__file__))
        model = curdir + '/../data/models/vanilla_S_0_CLF_2_R_none_photometry_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean_C.pt'
    else:
        # take the first element of the Series
        model = model[0]

    # add an exploded column with SNID
    df_tmp = pd.DataFrame.from_dict(
        {
            'jd': jd,
            'SNID': [str(i) for i in candid.values]
        }
    )

    df_tmp = df_tmp.explode('jd')

    # compute flux and flux error
    data = [mag2fluxcal_snana(*args) for args in zip(
        magpsf.explode(),
        sigmapsf.explode())]
    flux, error = np.transpose(data)

    # make a Pandas DataFrame with exploded series
    pdf = pd.DataFrame.from_dict({
        'SNID': df_tmp['SNID'],
        'MJD': df_tmp['jd'],
        'FLUXCAL': flux,
        'FLUXCALERR': error,
        'FLT': fid.explode().replace({1: 'g', 2: 'r'})
    })

    # Compute predictions
    ids, pred_probs = classify_lcs(pdf, model, 'cpu')

    # Reformat and re-index
    preds_df = reformat_to_df(pred_probs, ids=ids)
    preds_df.index = preds_df.SNID
    ia = preds_df.reindex([str(i) for i in candid.values])

    # return probabilities to be Ia
    return ia.prob_class0


if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    ztf_alert_sample = 'fink_science/data/alerts/alerts.parquet'
    globs["ztf_alert_sample"] = ztf_alert_sample

    model_path = 'fink_science/data/models/vanilla_S_0_CLF_2_R_none_photometry_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean_C.pt'
    globs["model_path"] = model_path

    # Run the test suite
    spark_unit_tests(globs)
