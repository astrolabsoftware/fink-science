# Copyright 2019 AstroLab Software
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
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import DoubleType

import pandas as pd
import numpy as np

from fink_science.active_learning_simple.classifier import fit_all_bands
from fink_science.active_learning_simple.classifier import load_external_model

from fink_science.tester import spark_unit_tests

@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def iaclassification(
        jd, fid, magpsf, sigmapsf, magnr,
        sigmagnr, magzpsci, isdiffpos, model) -> pd.Series:
    """ Return the probability of an alert to be a SNe Ia using a Random
    Forest Classifier.

    Parameters
    ----------
    jd: Spark DataFrame Column
        JD times (float)
    fid: Spark DataFrame Column
        Filter IDs (int)
    magpsf, sigmapsf: Spark DataFrame Columns
        Magnitude from PSF-fit photometry, and 1-sigma error
    magnr, sigmagnr: Spark DataFrame Columns
        Magnitude of nearest source in reference image PSF-catalog
        within 30 arcsec and 1-sigma error
    magzpsci: Spark DataFrame Column
        Magnitude zero point for photometry estimates
    isdiffpos: Spark DataFrame Column
        t => candidate is from positive (sci minus ref) subtraction
        f => candidate is from negative (ref minus sci) subtraction

    Returns
    ----------
    probabilities: 1D np.array of float
        Probability between 0 (non-Ia) and 1 (Ia).
    """
    # Compute the test_features: fit_all_bands
    test_features = fit_all_bands(
        jd.values, fid.values, magpsf.values, sigmapsf.values,
        magnr.values, sigmagnr.values, magzpsci.values, isdiffpos.values)

    # Load pre-trained model `clf`
    clf = load_external_model(model.values[0])

    # Make predictions
    probabilities = clf.predict_proba(test_features)

    # Check the type of prob
    # inverted wrt to original documentation: [pIa, pnon-Ia]
    to_return = probabilities.T[0]

    # Return probability of 0 for objects with no fit available (only zeros).
    mask = [np.sum(i) == 0.0 for i in test_features]
    to_return[mask] = 0.0
    # print(probabilities)

    return pd.Series(to_return)


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    spark_unit_tests(globals())