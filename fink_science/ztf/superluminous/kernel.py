# Copyright 2025 AstroLab Software
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

classifier_path = curdir + "/data/models/superluminous_classifier.joblib"
band_wave_aa = {1: 4770.0, 2: 6231.0, 3: 7625.0}
temperature = "sigmoid"
bolometric = "bazin"
min_points_total = 7
min_points_perband = 3
min_duration = 20
not_sl_threshold = -19.75
