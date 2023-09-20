from pyspark.sql.types import *

rate_module_output_schema = {
    # the first 5-sigma detection in the alert history
    "jd_first_real_det": DoubleType(),

    # The delta-time between the current jd of the alert and the jd_first_real_det,
    # if above 30, the jd_first_real_det is no longer reliable.
    "jdstarthist_dt": DoubleType(),

    # The magnitude rate computed between the last available measurement 
    # and the current alerts. The mag_rate band are the same of the current alert.
    "mag_rate": DoubleType(),

    # The magnitude rate error computed using a random sampling in the magnitude sigma of the two alerts.
    "sigma_rate": DoubleType(),

    # The lowest mag_rate computed from the random sampling
    "lower_rate": DoubleType(),

    # The highest mag_rate computed from the random sampling
    "upper_rate": DoubleType(),

    # The delta-time between the last alert and the current alert 
    "delta_time": DoubleType(),

    # if True, the magnitude rate has been computed with an upper limit.
    "from_upper": BooleanType()
}