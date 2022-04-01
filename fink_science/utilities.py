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

from fink_science.conversion import mag2fluxcal_snana
from fink_science.tester import regular_unit_tests

def concat_col(
        df, colname: str, prefix: str = 'c',
        current: str = 'candidate', history: str = 'prv_candidates'):
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
    current: str
        Name of the field containing current `colname` measurement, to extract
        `current.colname`. Usually a struct type field. Default is `candidate`
        from ZTF schema.
    history: str
        Name of the field containing history for `colname` measurements,
        to extract `history.colname`. Usually a list of struct type field.
        Default is `prv_candidates` from ZTF schema.

    Returns
    ----------
    df: DataFrame
        Dataframe with new column containing the concatenation of
        historical and current measurements.
    """
    return df.withColumn(
        prefix + colname,
        F.when(
            df['{}.{}'.format(history, colname)].isNotNull(),
            F.concat(
                df['{}.{}'.format(history, colname)],
                F.array(df['{}.{}'.format(current, colname)])
            )
        ).otherwise(F.array(df['{}.{}'.format(current, colname)]))
    )

def format_data_as_snana(
        jd, measurement, error, fid, candid, mask,
        filter_conversion_dic={1: 'g', 2: 'r'},
        transform_to_flux=True):
    """ Format data in SNANA units and format

    The resulting DataFrame is a concatenation of all alert data, with one
    measurement per row.

    |    a   |   b   |
    |--------|-------|
    | [1, 2] | [3, 4]|

    would become

    |    a   |   b   |
    |--------|-------|
    |    1   |   3   |
    |    2   |   4   |

    Parameters
    ----------
    jd: pd.Series
        Series containing Julian Dates (array of double). Each row contains
        all jd values for one alert (sorted).
    measurement: pd.Series
        Series containing data measurement (array of double). Each row contains
        all measurement values for one alert (sorted as jd).
        Can be either difference magnitude (a la ZTF), and which case you would
        set `transform_to_flux` to True to convert it into flux units,
        or it can be flux directly.
    error: pd.Series
        Series containing data error measurement (array of double). Each row
        contains all measurement values for one alert (sorted as jd).
        Can be either difference magnitude error (a la ZTF), and which case you
        would set `transform_to_flux` to True to convert it into flux units,
        or it can be flux error directly.
    fid: pd.Series
        Series containing filter band code (array of int/str). Each row contains
        all filter code values for one alert (sorted as jd).
    mask: pd.Series
        Series containing information on which alerts to keep (boolean).
    filter_conversion_dic: dict
        Mapping from telescope filter code (e.g. [1, 2] for ZTF) to
        SNANA filter code (['g', 'r']). Default is {1: 'g', 2: 'r'}.
    transform_to_flux: boolean
        Set it to True if `measurement` is in Difference magnitude
        units (default for ZTF), in which case we will convert to apparent
        magnitude and then SNANA flux. Default is True.

    Returns
    ----------
    pdf: pd.DataFrame
        DataFrame a la SNANA with SNID, MJD, FLUXCAL, FLUXCALERR, FLT.

    """
    # add an exploded column with SNID
    df_tmp = pd.DataFrame.from_dict(
        {
            'jd': jd[mask],
            'SNID': candid[mask]
        }
    )
    df_tmp = df_tmp.explode('jd')

    if transform_to_flux:
        # compute flux and flux error
        data = [mag2fluxcal_snana(*args) for args in zip(
            measurement[mask].explode(),
            error[mask].explode())]
        flux, flux_error = np.transpose(data)
    else:
        flux = measurement[mask].explode()
        flux_error = error[mask].explode()

    # make a Pandas DataFrame with exploded series
    pdf = pd.DataFrame.from_dict({
        'SNID': df_tmp['SNID'],
        'MJD': df_tmp['jd'].astype('float'),
        'FLUXCAL': flux.astype('float'),
        'FLUXCALERR': flux_error.astype('float'),
        'FLT': fid[mask].explode().replace(filter_conversion_dic)
    })

    return pdf

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


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    regular_unit_tests(globals())
