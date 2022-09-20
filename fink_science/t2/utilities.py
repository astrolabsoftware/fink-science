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
import os
import numpy as np
import pandas as pd

from tensorflow import keras

from astronet.metrics import WeightedLogLoss

from fink_science import __file__
from fink_utils.xmatch.simbad import return_list_of_eg_host

import tensorflow as tf

class LiteModel:
    @classmethod
    def from_file(cls, model_path):
        return LiteModel(tf.lite.Interpreter(model_path=model_path))

    @classmethod
    def from_saved_model(cls, model_path, tflite_file_path=None):
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
        ]
        converter.experimental_enable_resource_variables = True
        converter.experimental_new_converter = True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        if tflite_file_path is not None:
            with open(tflite_file_path, "wb") as f:
                f.write(tflite_model)

        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))

    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()[0]
        self.input_index = input_det["index"]
        self.output_index = output_det["index"]
        self.input_shape = input_det["shape"]
        self.output_shape = output_det["shape"]
        self.input_dtype = input_det["dtype"]
        self.output_dtype = output_det["dtype"]

    def predict(self, inp):
        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        out = np.zeros((count, self.output_shape[1]), dtype=self.output_dtype)
        for i in range(count):
            self.interpreter.set_tensor(self.input_index, inp[i: i + 1])
            self.interpreter.invoke()
            out[i] = self.interpreter.get_tensor(self.output_index)[0]
        return out

def get_lite_model(model_name: str = 'quantized-model-GR-noZ-28341-1654269564-0.5.1.dev73+g70f85f8-LL0.836.tflite'):
    path = os.path.dirname(__file__)
    model_path = (
        f"{path}/data/models/{model_name}"
    )
    model = LiteModel.from_file(model_path=model_path)
    return model

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
    path = os.path.dirname(__file__)
    model_path = (
        f"{path}/data/models/{model_name}/model-{model_id}"
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
