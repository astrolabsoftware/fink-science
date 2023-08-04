from fink_utils.spark.utils import concat_col
from pyspark.sql import functions as F
from fink_science.tester import add_roid_datatest
import os

from pyspark.sql import SparkSession
from pyspark import SparkConf

from fink_science.asteroids.processor import roid_catcher

from fink_science import __file__

spark = SparkSession.builder.getOrCreate()

conf = SparkConf()
confdic = {"spark.python.daemon.module": "coverage_daemon"}

if spark.version.startswith("2"):
    confdic.update(
        {
            "spark.jars.packages": "org.apache.spark:spark-avro_2.11:{}".format(
                spark.version
            )
        }
    )
elif spark.version.startswith("3"):
    confdic.update(
        {
            "spark.jars.packages": "org.apache.spark:spark-avro_2.12:{}".format(
                spark.version
            )
        }
    )
conf.setMaster("local[8]")
conf.setAppName("fink_science_test")

for k, v in confdic.items():
    conf.set(key=k, value=v)
spark = (
    SparkSession.builder.appName("fink_science_test").config(conf=conf).getOrCreate()
)

add_roid_datatest(spark, True)
path = os.path.dirname(__file__)
ztf_alert_sample = "file://{}/data/alerts/roid_datatest/alerts_sample_roid".format(path)
df = spark.read.load(ztf_alert_sample)

what = ["jd", "magpsf"]

prefix = "c"
what_prefix = [prefix + i for i in what]

for colname in what:
    df = concat_col(df, colname, prefix=prefix)

args = [
    "candidate.ra",
    "candidate.dec",
    "candidate.jd",
    "candidate.magpsf",
    "cjd",
    "cmagpsf",
    "candidate.fid",
    "candidate.ndethist",
    "candidate.sgscore1",
    "candidate.ssdistnr",
    "candidate.distpsnr1",
    F.lit(30),
    F.lit(2),
    F.lit(2),
    F.lit(30),
    F.lit(True),
]

df = df.withColumn("roid", roid_catcher(*args))

df = df.drop(*what_prefix)

# print(df.filter(df['roid.flag'] == 2).count())
# print(df.filter(df['roid.flag'] == 3).count())
# print(df.filter(df['roid.flag'] == 4).count())
# print(df.filter(df['roid.flag'] == 5).count())

# print()
# print()
# print()

# r = df.select("objectId", "roid.flag").toPandas()
# print(r.groupby("flag").count())

print()
print()
print()
print()
df_no_filt = df.select(df.columns + ["roid.flag", "roid.ffdistnr", "roid.estimator_id"])
print()
print()
# df_no_filt.explain(True)
print()
print()

pdf = df_no_filt.toPandas()
r = pdf.explode(["ffdistnr", "estimator_id"])

print(r[(r["flag"] == 4) | (r["flag"] == 5)])

print()
print()
print()
print()
print()
print()

df_filt = df.filter((df["roid.flag"] == 4) | (df["roid.flag"] == 5)).select(
    df.columns + ["roid.flag", "roid.ffdistnr", "roid.estimator_id"]
)
print()
print()
# df_filt.explain(True)
print()
print()

pdf2 = df_filt.toPandas()
r = pdf2.explode(["ffdistnr", "estimator_id"])

print(r)
print()
print()

# print(pdf[pdf["flag"] == 5])

# print()
# print()

# print(pdf[pdf["flag"] == 4])

# print()
# print()

# print(df.select(df.columns + ["roid.flag", "roid.ffdistnr", "roid.estimator_id"]).filter(F.col("flag") == '5').toPandas())

# print()
# print()

# print(df.select(df.columns + ["roid.flag", "roid.ffdistnr", "roid.estimator_id"]).filter(F.col("flag") == '4').toPandas())
