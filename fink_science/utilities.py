# Copyright 2020-2021 AstroLab Software
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
from pyspark.sql import functions as F

import pandas as pd
import numpy as np
import pickle

from fink_science.tester import regular_unit_tests

def concat_col(df, colname: str, prefix: str = 'c'):
    """ Add new column to the DataFrame named `prefix`+`colname`, containing
    the concatenation of historical and current measurements.

    Parameters
    ----------
    df: DataFrame
        Pyspark DataFrame containing alert data
    colname: str
        Name of the column to add (without the prefix)
    prefix: str
        Additional prefix to add to the column name. Default is 'c'.

    Returns
    ----------
    df: DataFrame
        Dataframe with new column containing the concatenation of
        historical and current measurements.
    """
    return df.withColumn(
        prefix + colname,
        F.when(
            df['prv_candidates.{}'.format(colname)].isNotNull(),
            F.concat(
                df['prv_candidates.{}'.format(colname)],
                F.array(df['candidate.{}'.format(colname)])
            )
        ).otherwise(F.array(df['candidate.{}'.format(colname)]))
    )

def extract_field(current: list, history: list) -> np.array:
    """ Concatenate current and historical data.

    If t1 is the first time the object has been seen, and the object has N
    historical measurements, the routine returns values ordered as:
    [t1, t2, ...., tN, current] (past to current).

    Parameters
    ----------
    current: list [nalert, 1]
        List of field values. each entry corresponds to the measurement for
        one alert.
    history: list of list [nalerts, Ndays]
        List of historical field values. Each entry is a list of historical
        measurements for one alert.

    Returns
    ----------
    conc: 2D np.array [nalert, Ndays + 1]
        Array of array. Each entry is an array of historical+current
        measurements for one alert.

    Examples
    ----------
    >>> current = [1, 1]
    >>> historical = [[4, 3, 2], [4, 3, 2]]
    >>> c = extract_field(current, historical)
    >>> print(c) # doctest: +NORMALIZE_WHITESPACE
    [[4 3 2 1] [4 3 2 1]]
    """
    conc = [np.concatenate((j, [i])) for i, j in zip(current, history)]
    return np.array(conc)

def load_scikit_model(fn: str = ''):
    """ Load a RandomForestClassifier model from disk (pickled).

    Parameters
    ----------
    fn: str
        Filename. This file should be known from all machines!

    Return
    ----------
    clf: sklearn.ensemble.forest.RandomForestClassifier

    Examples
    >>> fn = 'fink_science/data/models/default-model_bazin.obj'
    >>> model = load_scikit_model(fn)
    >>> 'RandomForestClassifier' in str(type(model))
    True

    # binary classification
    >>> model.n_classes_
    2
    """
    return pickle.load(open(fn, 'rb'))

def load_pcs(fn, npcs):
    """ Load PC from disk into a Pandas DataFrame

    Parameters
    ----------
    fn: str
        Filename. This file should be known from all machines!
    npcs: int
        Number of principal components to load

    Return
    ----------
    clf: sklearn.ensemble.forest.RandomForestClassifier
    """
    comp = pd.read_csv(fn)
    pcs = {}
    for i in range(npcs):
        pcs[i + 1] = comp.iloc[i].values

    return pd.DataFrame(pcs)


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    regular_unit_tests(globals())
