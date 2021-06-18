# Copyright 2021 AstroLab Software
# Author: Roman Le Montagner
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
from fink_science.image_classification.image_classification import img_labelisation
import pandas as pd
import os
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StringType
from fink_science import __file__
from fink_science.tester import spark_unit_tests

@pandas_udf(StringType(), PandasUDFType.SCALAR)
def labels_assignation(stamps: bytes) -> pd.Series:
    """ Apply image classification on column of images

    Parameters
    ----------
    stamps: Spark DataFrame column
        column which contains image in gzip format file

    Returns
    -------
    out: pd.Series
        a pandas series which contains all images labels

    Examples
    --------
    >>> df = spark.read.format('parquet').load(ztf_alert_sample)
    >>> img_classification_df = df.withColumn('labels', labels_assignation(df['cutoutScience.stampData'])).toPandas()

    >>> safe_clear_star = img_classification_df[img_classification_df['objectId'] == 'ZTF18acrunkm']
    >>> list(safe_clear_star['labels'] == 'safe_clear_star')[0] == True
    True

    >>> safe_clear_star = img_classification_df[img_classification_df['objectId'] == 'ZTF20aafdzuq']
    >>> list(safe_clear_star['labels'] == 'safe_noisy')[0] == True
    True

    >>> safe_clear_star = img_classification_df[img_classification_df['objectId'] == 'ZTF18aabipja']
    >>> list(safe_clear_star['labels'] == 'corrupted_clear')[0] == True
    True

    >>> safe_clear_star = img_classification_df[img_classification_df['objectId'] == 'ZTF18abuajuu']
    >>> list(safe_clear_star['labels'] == 'safe_clear_extend')[0] == True
    True
    """
    res = [img_labelisation(cutout) for cutout in stamps.values]

    return pd.Series(res)


if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = 'file://{}/data/alerts/image_classification_example.parquet'.format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
