from classifier import agn_classifier
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
import pandas as pd
import os
from fink_science import __file__
from fink_science.tester import spark_unit_tests


@pandas_udf(DoubleType())
def agn_spark(objectId, jd, magpsf, sigmapsf, fid):

    """
    Examples
    --------
    >>> df = spark.read.format('parquet').load(ztf_alert_sample)
    >>> df_agn = df.withColumn('proba', agn_spark(df.objectId, df.cjd, df.cmagpsf, df.csigmapsf, df.cfid))
    >>> df_agn.show()

    """

    data = pd.DataFrame(
        {
            "objectId": objectId,
            "cjd": jd,
            "cmagpsf": magpsf,
            "csigmapsf": sigmapsf,
            "cfid": fid,
        }
    )

    proba = agn_classifier(data)
    return pd.Series(proba)


if __name__ == "__main__":

    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = "file://{}/data/alerts/agn_example.parquet".format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)