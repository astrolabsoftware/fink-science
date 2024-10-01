# Copyright 2022 Fink Software
# Author: Etienne Russeil
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
from fink_science import __file__

curdir = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_ELASTICC_WITH_MD = curdir + "/data/models/SLSN_rainbow_MD.joblib"
CLASSIFIER_ELASTICC_NO_MD = curdir + "/data/models/SLSN_rainbow_no_MD.joblib"
MINIMUM_POINTS = 7
PASSBANDS = ["u", "g", "r", "i", "z", "Y"]
MAXFEV = 500
