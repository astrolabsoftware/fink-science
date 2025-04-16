# Copyright 2019-2022 AstroLab Software
# Authors: Marina Masson
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
import pandas as pd
import numpy as np
import joblib

from fink_science.orphans.classifier import orphan_classifier

from fink_filters.tester import spark_unit_tests

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import FloatType

@pandas_udf(FloatType())
def orphan_grb(ctimemjd, cabmags, cabmagserr, cfilts):
    """ Filter to extract orphan GRB candidates

    The filter is based on different features calculated from the light curve:
    - Duration between the first detection and the peak
    - Increase and decrease rates
    - Colour between different filter pairs
    - 4 parameters and chi2 from fitting the light curve

    A Boosting Decision Tree classifier was trained to discriminate orphans using these features

    Parameters
    ----------
    ctimemjd: pandas.Series of list of float
        Concatenated time in MJD for the object
    cabmags: pandas.Series of list of float, or Nan
        Concatenated magnitude for the object
    cabmagserr: pandas.Series of list of float, or Nan
        Concatenated errors on the magnitude for the object
    cfilts: pandas.Series of list of int
        Concatenated filters for the object

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the proba of an event to be an orphan
    """

    valid = cabmags.apply(lambda x: True if len(~np.isnan(x)) >= 5 else False)
    proba_orphan = orphan_classifier(ctimemjd, cabmags, cabmagserr, cfilts, valid)

    return pd.Series(proba_orphan)


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)