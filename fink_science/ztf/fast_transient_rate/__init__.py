# Copyright 2020-2023 AstroLab Software
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
from pyspark.sql.types import DoubleType, BooleanType

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
    "from_upper": BooleanType(),
}
