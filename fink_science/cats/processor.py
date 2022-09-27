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
from pyspark.sql.types import StructType, StructField, ArrayType, FloatType

import tensorflow as tf
from tensorflow_addons import optimizers

from fink_science import __file__
from fink_science.cats.utilities import normalize_lc, extract_max_prob
from fink_science.tester import spark_unit_tests

tf.optimizers.RectifiedAdam = optimizers.RectifiedAdam


@pandas_udf(StructType([
    StructField("broad_preds", ArrayType(FloatType())),
    StructField("fine_preds", ArrayType(FloatType()))
]), PandasUDFType.SCALAR)
def predict_nn(
        midpointTai: pd.Series, psFlux: pd.Series, psFluxErr: pd.Series,
        filterName: pd.Series, mwebv: pd.Series, z_final: pd.Series,
        z_final_err: pd.Series, hostgal_zphot: pd.Series,
        hostgal_zphot_err: pd.Series,
        model=None
) -> pd.DataFrame:
    """ Return predctions from a CBPF hierarchical model using Elasticc alert data
    For the default model, one has:
    class_dict = {
        0: 11,
        1: 12,
        2: 13,
        3: 21,
        4: 22,
    }
    For the fine classifiers mapping, one has:
    fine_classifier_map = {
        0: sn_model,
        1: fast_model,
        2: long_model,
        3: periodic_model,
        4: None
    }
    Each of the fine classifiers has it's own mapping:
    SN model:
        SN_map = {
            0: 'SNIa',
            1: 'SNIb/c',
            2: 'SNII',
            3: 'SNIax',
            4: 'SNIa-91bg'
        }
    Fast model:
        fast_model = {
            0: 'KN',
            1: 'M-dwarf-flare',
            2: 'dwarf novae',
            3: 'uLens'
        }
    Long model:
        long_model = {
            0: 'SLSN',
            1: 'TDE',
            2: 'ILOT',
            3: 'CART',
            4: 'PISN'
        }
    Periodic model:
        periodic_model = {
            0: 'Cepheid',
            1: 'RR Lyrae',
            2: 'Delta Scuti',
            3: 'EB'
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
    preds_df: pd.DataFrame
        preds_df is an pd.DataFrame which contains a 'broad_class' column
        with probabilities for broad classes (e.g. SN-like, Fast, Long,
        Periodic and AGN) and a second column called 'fine_preds' with
        predictions for finer_classes according to each fine classifier.
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
    >>> df = df.withColumn('cbpf_class', F.col('preds.broad_preds').getItem(0).astype('int'))
    >>> df = df.withColumn('cbpf_max_prob', F.col('preds.broad_preds').getItem(1))
    >>> df.filter(df['cbpf_class'] == 0).count()
    39
    """

    filter_dict = {'u': 1, 'g': 2, 'r': 3, 'i': 4, 'z': 5, 'Y': 6}

    curdir = os.path.dirname(os.path.abspath(__file__))
    models_path = curdir + '/data/models/cats_models/'

    sn_model = tf.keras.models.load_model(
        models_path + '/model_test_meta_ragged_1det_sn_tuner_1month.h5',
        custom_objects={'RectifiedAdam': optimizers.RectifiedAdam}
    )
    fast_model = tf.keras.models.load_model(
        models_path + '/model_test_meta_ragged_1det_fast_tanh.h5',
        custom_objects={'RectifiedAdam': optimizers.RectifiedAdam}
    )

    long_model = tf.keras.models.load_model(
        models_path + '/model_test_meta_ragged_1det_long_tuner_1month.h5',
        custom_objects={'RectifiedAdam': optimizers.RectifiedAdam}
    )

    periodic_model = tf.keras.models.load_model(
        models_path + '/model_test_meta_ragged_1det_periodic_tuner_1month.h5',
        custom_objects={'RectifiedAdam': optimizers.RectifiedAdam}
    )

    fine_classifier_map = {
        0: sn_model,
        1: fast_model,
        2: long_model,
        3: periodic_model,
        4: None
    }

    bands = []
    lcs = []
    meta = []

    for i, mjds in enumerate(midpointTai):

        if len(mjds) > 0:
            bands.append(np.array(
                [filter_dict[f] for f in filterName.values[i]]
            ).astype(np.int16))
            lc = np.concatenate(
                [mjds[:, None], psFlux.values[i][:, None],
                    psFluxErr.values[i][:, None]], axis=-1
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
        'band': tf.RaggedTensor.from_row_lengths(
            values=tf.concat(bands, axis=0),
            row_lengths=[a.shape[0] for a in bands]
        ),

        'lc': tf.RaggedTensor.from_row_lengths(
            values=tf.concat(lcs, axis=0),
            row_lengths=[a.shape[0] for a in lcs]
        )
    }
    for i, x in enumerate(X['meta'][:, 3]):
        if x < 0:
            X['meta'][i, 1:] = -1
        else:
            X['meta'][i, 1:] = x

    if model is None:
        # Load pre-trained model
        curdir = os.path.dirname(os.path.abspath(__file__))
        model_path = curdir + '/data/models/cats_models/model_test_meta_ragged_1det_broad_tuner.h5'
    else:
        model_path = model.values[0]

    NN = tf.keras.models.load_model(
        model_path, custom_objects={
            'RectifiedAdam': optimizers.RectifiedAdam
        })
    preds = NN.predict(X)
    preds_fine = []

    for i, p in enumerate(preds):
        if np.nan not in p:
            if p.argmax() <= 3:
                pred = fine_classifier_map[p.argmax()].predict(
                    [X['band'][i:i + 1], X['lc'][i:i + 1], X['meta'][i:i + 1]])
                if pred.shape[1] == 4:
                    pred = np.concatenate((pred, [[-1.0]]), axis=1)

                preds_fine.append(pred)

            else:
                to_concat = np.array([[p[-1]]])
                pred = np.concatenate((to_concat, [[-1.0] * 4]), axis=1)
                preds_fine.append(pred)
        else:
            preds_fine.append(np.array([[-1.0] * 5]))

    preds_fine = np.concatenate(preds_fine)

    preds_df = pd.DataFrame(
        {
            'broad_preds': [extract_max_prob(elem) for elem in preds],
            'fine_preds': [extract_max_prob(elem) for elem in preds_fine]
        }
    )

    return preds_df


if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    path = os.path.dirname(__file__)

    elasticc_alert_sample = 'file://{}/data/alerts/elasticc_sample_seed0.parquet'.format(path)
    globs["elasticc_alert_sample"] = elasticc_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
