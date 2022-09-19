# Copyright 2022 AstroLab Software
# Author: Tarek Allam, Julien Peloton
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
import numpy as np
import pandas as pd

from tensorflow import keras

from astronet.metrics import WeightedLogLoss

from fink_science import __file__
from fink_utils.xmatch.simbad import return_list_of_eg_host

def get_model(model_name: str = 't2', model_id: str = "23057-1642540624-0.1.dev963+g309c9d8"):
    """ Load pre-trained model for T2

    Parameters
    ----------
    model_name: str
        Folder name containing pre-trained models. Available: t2, atx
    model_id: str
        Corresponding ID inside the foler (related to the version used to train)

    Returns
    ----------
    out: keras model
    """
    model_path = (
        f"{__file__}/data/models/{model_name}/model-{model_id}"
    )

    model = keras.models.load_model(
        model_path,
        custom_objects={"WeightedLogLoss": WeightedLogLoss()},
        compile=False,
    )

    return model

def apply_selection_cuts_ztf(
        magpsf: pd.Series, cdsxmatch: pd.Series,
        jd: pd.Series, jdstarthist: pd.Series, roid: pd.Series,
        minpoints: int = 2, maxndethist: int = 90) -> pd.Series:
    """ Apply selection cuts to keep only alerts of interest
    for T2 analysis

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

    # Add cuts on having exactly 2 filters

    return mask
