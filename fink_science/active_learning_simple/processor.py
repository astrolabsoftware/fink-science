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
from pyspark.sql.types import BooleanType, DoubleType

import pandas as pd
import numpy as np

from fink_science.active_learning_simple.classifier import apply_classifier
from fink_science.active_learning_simple.classifier import extract_field

from fink_science.tester import spark_unit_tests

from typing import Any

@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def iaclassification(time, mag, band, htimes, hmags, hbands) -> pd.Series:
    """ Return the probability of an alert to be a SN1a using a Random
    Forest Classifier.

    Parameters
    ----------
    time: Spark DataFrame Column
        Column of candidate time (float): candidate.jd
    mag: Spark DataFrame Column
        Column of candidate flux (float): candidate.magpsf
    band: Spark DataFrame Column
        Column of candidate filter ID (int): candidate.fid
    htimes: Spark DataFrame Column
        Column of historical time vectors: prv_candidates.jd
    hmags: Spark DataFrame Column
        Column of historical flux vectors: prv_candidates.magpsf
    hbands: Spark DataFrame Column
        Column of historical filter ID vectors: prv_candidates.fid
    """
    alltimes = extract_field(time, htimes)
    allmags = extract_field(mag, hmags)
    allbands = extract_field(band.values, hbands.values)

    predictions, probabilities = apply_classifier(
        alltimes, allmags, allbands
    )

    # print(predictions, probabilities)

    # Check the type of prob
    return pd.Series(probabilities.T[0])


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    spark_unit_tests(globals())
