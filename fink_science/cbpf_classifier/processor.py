import tensorflow as tf
import numpy as np
from utilities import normalize_lc
import pandas as pd
from sklearn.metrics import confusion_matrix

def predict_nn(
    mid_point_tai,
    ps_flux,
    ps_flux_err,
    filter_name,
    mwebv,
    z_final,
    z_final_err,
    hostgal_zphot,
    hostgal_zphot_err,
    model
    ) -> pd.Series:
    """
    Return predctions from a model given inputs

    Parameters
    ----------
    mid_point_tai:
    
    ps_flux:

    ps_flux_err:

    filter_name:

    mwebv:

    z_final:

    z_final_err:

    hostgal_zphot:

    hostgal_zphot_err:

    model: str

    Returns
    -------
    out: tf.keras.Model
        Returns an instance of tf.keras.Model class.
    """
    filter_dict = {'u':1, 'g':2, 'r':3, 'i':4, 'z':5, 'Y':6}
    bands = []
    lcs = []
    meta = []
    for i, mjds in enumerate(mid_point_tai):
        bands.append(np.array([filter_dict[f] for f in filter_name[i]]).astype(np.int16))
        lc = np.concatenate(
            [mjds[:,None], ps_flux[i][:,None], ps_flux_err[:,None]], axis=-1
            )
        lcs.append(normalize_lc(lc).astype(np.float32))
        meta.append(
            np.concatenate(
                [mwebv[i], z_final[i], z_final_err[i], hostgal_zphot[i], hostgal_zphot_err[i]]
            )
        )

    X = {
        'meta': np.array(meta),
        'band': tf.RaggedTensor.from_row_lengths(values=tf.concat(bands, axis=0), row_lengths=[a.shape[0] for a in bands]),
        'lc': tf.RaggedTensor.from_row_lengths(values=tf.concat(lcs, axis=0), row_lengths=[a.shape[0] for a in lcs])
    }
    for i, x in enumerate(X['meta'][:,3]):
        if x < 0:
            X['meta'][i,1:] = -1
        else:
            X['meta'][i,1:] = x

    nn = tf.keras.models.load_model(model)
    preds = nn.predict(X)
    return pd.Series(preds)
