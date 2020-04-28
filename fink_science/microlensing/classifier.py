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
import pickle
import os

from pyspark.sql.types import StructType, StructField, StringType, DoubleType

def load_mulens_schema_twobands():
    """ DataFrame Schema for the mulens UDF using 2 filter bands

    Returns
    --------
    schema: StructType
        StructType with StructFields describing new columns to be added
        when using `mulens` in pyspark UDF. There are 2 new columns per
        filter bands: the classification name (str) and
        the mulens score (double).
    """
    schema = StructType([
        StructField("class_1", StringType(), True),
        StructField("ml_score_1", DoubleType(), True),
        StructField("class_2", StringType(), True),
        StructField("ml_score_2", DoubleType(), True)
    ])
    return schema

def load_external_model(model_path):
    """ Unpickle pre-loaded models from LIA.

    Procedure to train the models can be found at:
    https://github.com/tblaineau/ZTF_mulens_simulator

    Parameters
    ----------
    model_path: str
        Folder that contains the two models `rf.sav` and `pca.sav`.

    Returns
    ----------
    rf, pca: RandomForestClassifiers
    """
    rf = pickle.load(open(os.path.join(model_path, 'rf.sav'), 'rb'))
    pca = pickle.load(open(os.path.join(model_path, 'pca.sav'), 'rb'))
    return rf, pca
