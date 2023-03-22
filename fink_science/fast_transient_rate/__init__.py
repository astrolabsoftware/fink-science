from pyspark.sql.types import *

rate_module_output_schema = {
    "jd_first_real_det": DoubleType(),
    "jdstarthist_dt": DoubleType(),
    "mag_rate": DoubleType(),
    "sigma_rate": DoubleType(),
    "lower_rate": DoubleType(),
    "upper_rate": DoubleType(),
    "delta_time": DoubleType(),
    "from_upper": BooleanType()
}