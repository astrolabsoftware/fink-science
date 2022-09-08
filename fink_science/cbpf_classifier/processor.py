from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import DoubleType

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow_addons import optimizers
from utilities import normalize_lc

tf.optimizers.RectifiedAdam = optimizers.RectifiedAdam

@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def predict_nn(
        midpointTai: pd.Series, psFlux: pd.Series, psFluxErr: pd.Series,
        filterName: pd.Series, mwebv: pd.Series, z_final: pd.Series,
        z_final_err: pd.Series, hostgal_zphot: pd.Series,
        hostgal_zphot_err: pd.Series,
        model
        ) -> pd.Series:
    """
    Return predctions from a model given inputs as pd.Series

    Parameters:
    -----------
    midpointTai: spark DataFrame Column
        SNID JD Time (float)
    psFlux: spark DataFrame Column
        flux from LSST (float)
    psFluxErr: spark DataFrame Column
        flux error from LSST (float)
    filterName:
        (string)
    mwebv:
        (float)
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
        predictions of a broad class in an pd.Series format (pd.Series[float])
    """

    filter_dict = {'u':1, 'g':2, 'r':3, 'i':4, 'z':5, 'Y':6}
    bands = []
    lcs = []
    meta = []

    for i, mjds in enumerate(midpointTai):
        bands.append(np.array(
            [filter_dict[f] for f in filterName[i]]
        ).astype(np.int16))        
        lc = np.concatenate(
            [mjds[:,None], psFlux[i][:,None], psFluxErr[:,None]], axis=-1
            )
        lcs.append(normalize_lc(lc).astype(np.float32))
        meta.append(
            np.concatenate(
                [mwebv[i], z_final[i], z_final_err[i], hostgal_zphot[i], hostgal_zphot_err[i]]
            )
        )

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
    for i, x in enumerate(X['meta'][:,3]):
        if x < 0:
            X['meta'][i,1:] = -1
        else:
            X['meta'][i,1:] = x

    NN = tf.keras.models.load_model(model)
    preds = NN.predict(X)
    
    return pd.Series(preds)
