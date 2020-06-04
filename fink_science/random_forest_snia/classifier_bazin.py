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
import numpy as np

from fink_science.random_forest_snia.bazin import fit_scipy
from fink_science.conversion import mag2fluxcal_snana

from fink_science.tester import regular_unit_tests

def fit_all_bands(jd, fid, magpsf, sigmapsf) -> np.array:
    """ Perform a Bazin fit for all alerts and all bands.

    For a given set of parameters (a, b, ...), and a given
    set of bands (g, r, ...), the final feature vector must be of the form:

    features = [
        [ga, gb, ... ra, rb, ... ], # alert 0
        [ga, gb, ... ra, rb, ... ], # alert 1
        [ga, gb, ... ra, rb, ... ], # alert ...
    ]

    Parameters
    ----------
    in: 2D np.array (alerts, time-series)
        Array of property vectors (float)

    Returns
    ----------
    features: 2D np.array (alerts, features x bands)
        Array of feature vectors (all bands for each alert)
    """
    features = []
    unique_bands = [1, 2, 3]
    # Loop over all alerts
    for alert_data in zip(jd, fid, magpsf, sigmapsf):
        (ajd, afid, amagpsf, asigmapsf) = alert_data

        feature_alert = []
        # For each alert, estimate the parameters for each band
        for band in unique_bands:
            maskband = afid == band
            masknan = amagpsf == amagpsf
            masknone = amagpsf != None
            mask = maskband * masknan * masknone
            if ajd is None or len(ajd[mask]) < 5:
                # Not sure what is going on in this case
                feature_alert.extend(np.zeros(5, dtype=np.float))
            else:
                # Compute flux
                flux, sigmaflux = mag2fluxcal_snana(
                    amagpsf[mask], asigmapsf[mask])
                feature_alert.extend(fit_scipy(ajd[mask], flux))
        features.append(np.array(feature_alert))
    return np.array(features)


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    regular_unit_tests(globals())
