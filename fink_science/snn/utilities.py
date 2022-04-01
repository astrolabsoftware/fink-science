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
import numpy as np
import pandas as pd

from fink_science.tester import regular_unit_tests

def reformat_to_df(pred_probs, ids=None):
    """ Reformat supernnova predictions to DataFrame.

    Taken from Anais's SuperNNova.

    Parameters
    ----------
    pred_probs: list of list
        contains list of probabilities for each alert:
        [
            [prob_class0_alert0, prob_class1_alert0],
            [prob_class0_alert1, prob_class1_alert1],
            ...
        ]
    ids: list of str
        List of intial alert SNID

    Returns
    ----------
    preds_df: pd.DataFrame
        Pandas DataFrame, row={SNID  prob_class0  prob_class1  pred_class}
    """
    # TO DO: suppport nb_inference != 1
    num_inference_samples = 1

    d_series = {}
    for i in range(pred_probs[0].shape[1]):
        d_series["SNID"] = []
        d_series[f"prob_class{i}"] = []
    for idx, value in enumerate(pred_probs):
        d_series["SNID"] += [ids[idx]] if len(ids) > 0 else idx
        value = value.reshape((num_inference_samples, -1))
        value_dim = value.shape[1]
        for i in range(value_dim):
            d_series[f"prob_class{i}"].append(value[:, i][0])
    preds_df = pd.DataFrame.from_dict(d_series)

    # get predicted class
    preds_df["pred_class"] = np.argmax(pred_probs, axis=-1).reshape(-1)

    return preds_df


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    regular_unit_tests(globals())
