# Copyright 2020-2022 AstroLab Software
# Author: Andre Santos, Bernardo Fraga, Clecio de Bom
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

import os
import numpy as np
import pandas as pd

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, FloatType

import tensorflow as tf
from tensorflow_addons import optimizers

from fink_science import __file__
from fink_science.cats.utilities import normalize_lc, extract_max_prob
from fink_science.tester import spark_unit_tests

tf.optimizers.RectifiedAdam = optimizers.RectifiedAdam


@pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR)
def predict_nn(
        midpointTai: pd.Series, psFlux: pd.Series, psFluxErr: pd.Series,
        filterName: pd.Series, mwebv: pd.Series, z_final: pd.Series,
        z_final_err: pd.Series, hostgal_zphot: pd.Series,
        hostgal_zphot_err: pd.Series,
        model=None
) -> pd.DataFrame:
    """ Return predctions from a CBPF classifier model (cats general) using Elasticc alert data.
    For the default model, one has the following mapping:

    class_dict = {
        0: 111,
        1: 112,
        2: 113,
        3: 114,
        4: 115,
        5: 121,
        6: 122,
        7: 123,
        8: 124,
        9: 131,
        10: 132,
        11: 133,
        12: 134,
        13: 135,
        14: 211,
        15: 212,
        16: 213,
        17: 214,
        18: 221
    }

    Parameters:
    -----------
    midpointTai: spark DataFrame Column
        SNID JD Time (float)
    psFlux: spark DataFrame Column
        flux from LSST (float)
    psFluxErr: spark DataFrame Column
        flux error from LSST (float)
    filterName: spark DataFrame Column
        observed filter (string)
    mwebv: spark DataFrame Column
        milk way extinction (float)
    z_final: spark DataFrame Column
        redshift of a given event (float)
    z_final_err: spark DataFrame Column
        redshift error of a given event (float)
    hostgal_zphot: spark DataFrame Column
        photometric redshift of host galaxy (float)
    hostgal_zphot_err: spark DataFrame Column
        error in photometric redshift of host galaxy (float)
    model: spark DataFrame Column
        path to pre-trained Hierarchical Classifier model. (string)

    Returns:
    --------
    preds: pd.Series
        preds is an pd.Series which contains a 'cats_general_preds' column
        with probabilities for classes shown in Elasticc data challenge.

    Examples
    -----------
    >>> from fink_utils.spark.utils import concat_col
    >>> from pyspark.sql import functions as F
    >>> df = spark.read.format('parquet').load(elasticc_alert_sample)

    # Assuming random positions
    >>> df = df.withColumn('cdsxmatch', F.lit('Unknown'))
    >>> df = df.withColumn('roid', F.lit(0))

    # Required alert columns
    >>> what = ['midPointTai', 'psFlux', 'psFluxErr', 'filterName']

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
    >>> args += [F.col('diaObject.mwebv'), F.col('diaObject.z_final'), F.col('diaObject.z_final_err')]
    >>> args += [F.col('diaObject.hostgal_zphot'), F.col('diaObject.hostgal_zphot_err')]
    >>> df = df.withColumn('preds', predict_nn(*args))
    >>> df = df.withColumn('cbpf_class', F.col('preds').getItem(0).astype('int'))
    >>> df = df.withColumn('cbpf_max_prob', F.col('preds').getItem(1))
    >>> df.filter(df['cbpf_class'] == 0).count()
    0
    """

    filter_dict = {'u': 1, 'g': 2, 'r': 3, 'i': 4, 'z': 5, 'Y': 6}

    bands = []
    lcs = []
    meta = []
    frac = 10**(- (31.4 - 27.5) / 2.5)

    for i, mjds in enumerate(midpointTai):

        if len(mjds) > 0:
            bands.append(np.array(
                [filter_dict[f] for f in filterName.values[i]]
            ).astype(np.int16))
            lc = np.concatenate(
                [
                    mjds[:, None],
                    frac * psFlux.values[i][:, None],
                    frac * psFluxErr.values[i][:, None]
                ], axis=-1
            )

            if not np.isnan(mwebv.values[i]):

                lcs.append(normalize_lc(lc).astype(np.float32))

                meta.append([
                    mwebv.values[i], z_final.values[i],
                    z_final_err.values[i], hostgal_zphot.values[i],
                    hostgal_zphot_err.values[i]
                ])

    X = {
        'meta': np.array(meta),
        'band': tf.keras.preprocessing.sequence.pad_sequences(bands,
                                                              maxlen=243,
                                                              dtype='int32'),
        'lc': tf.keras.preprocessing.sequence.pad_sequences(lcs,
                                                            maxlen=243,
                                                            dtype='float32'),
    }

    for i, x in enumerate(X['meta'][:, 3]):
        if x < 0:
            X['meta'][i, 1:] = -1
        else:
            X['meta'][i, 1:] = x

    if model is None:
        # Load pre-trained model
        curdir = os.path.dirname(os.path.abspath(__file__))
        model_path = curdir + '/data/models/cats_models/model_test_meta_alerts_tuner.h5'
    else:
        model_path = model.values[0]

    NN = tf.keras.models.load_model(
        model_path, custom_objects={
            'RectifiedAdam': optimizers.RectifiedAdam
        })

    preds = NN.predict(X)

    return pd.Series([extract_max_prob(elem) for elem in preds])


if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    path = os.path.dirname(__file__)

    elasticc_alert_sample = 'file://{}/data/alerts/elasticc_sample_seed0.parquet'.format(
        path)
    globs["elasticc_alert_sample"] = elasticc_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
