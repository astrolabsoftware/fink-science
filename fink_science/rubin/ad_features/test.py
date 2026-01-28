# Copyright 2019-2026 AstroLab Software
# Author: Timofei Pshenichnyy, Matwey V. Kornilov
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
import numpy as np

from fink_science.rubin.ad_features.processor import (
    extract_features_ad_rubin_raw,
    FEATURES_COLS,
)

# We need a normal test here, but I don't know how to do it yet
np.random.seed(42)
N = 100
mjds = np.sort(np.random.uniform(59000, 60000, N))
flux = np.random.normal(1000, 100, N)  # fake nJy
fluxerr = np.random.uniform(5, 20, N)
# Randomly assign bands g and r
bands = np.random.choice(["g", "r"], N)

# Test raw function
features = extract_features_ad_rubin_raw(mjds, flux, fluxerr, bands, "TestObject1")

# Basic assertions
assert len(features) == 2, "Should have features for 2 bands (g, r)"
assert "g" in features, "Band g missing"
assert "r" in features, "Band r missing"
assert len(features["g"]) == len(FEATURES_COLS), (
    "Incorrect number of features extracted"
)

print("Test passed: Features extracted successfully.")
print(f"Example feature (Amplitude band g): {features['g']['amplitude']}")
