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

"""Configuration constants for the superluminous supernovae (SLSN) classifier.

This module centralises the paths, physical constants and cuts shared by
:mod:`fink_science.ztf.superluminous.processor` and
:mod:`fink_science.ztf.superluminous.slsn_classifier`, so that the
classifier's behaviour can be tuned from a single place.
"""

import os
from fink_science import __file__

curdir = os.path.dirname(os.path.abspath(__file__))

# Path to the pre-trained classifier (scikit-learn compatible pipeline,
# pickled with joblib). It is expected to expose `predict_proba`,
# `feature_names_in_` and `optimal_threshold` attributes.
classifier_path = curdir + "/data/models/NOMAI.joblib"

# Path to the REGALADE galaxy catalog (Tranin et al., see
# https://github.com/htranin/regalade), reduced to the columns needed for
# host crossmatch and photo-z (gal_ra, gal_dec, R1, R2, PA, z, z_err) and to
# the ZTF-visible sky (dec > -30). Not distributed with the package (~1.5GB,
# far above what can be committed to git or bundled in a pip package) --
# see fink_science/data/catalogs/README.md for how to obtain it.
regalade_path = curdir + "/data/catalogs/regalade_minimal_ZTF.fits"

# Directional light radius (DLR) scale factor applied to the REGALADE
# ellipses before host crossmatch (see `slsn_classifier.get_regalade_photoz`):
# grows every galaxy's ellipse to avoid missing transients hosted with large
# offsets. Same value used to build the training set, see
# `create_photoz_table.py` in the training pipeline.
regalade_dlr_factor = 1.25

# Number of geomspace bins (by ellipse semi-major axis R1) used to speed up
# the REGALADE host crossmatch on the ~56M-galaxy catalog. Same value used
# to build the training set.
regalade_nbins = 20

# Effective wavelength (in Angstrom) of the ZTF filters, keyed by their
# `fid`/`cfid` integer code (1=g, 2=r). Used both to fit the Rainbow model
# (`fit_rainbow`) and to compute Milky Way extinction and absolute
# magnitudes (`abs_peak`). i-band (fid=3) is intentionally excluded: it was
# not used to train the classifier, see `remove_bad_bands`.
# Source: http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=Palomar&gname2=ZTF&asttype=
band_wave_aa = {1: 4746.48, 2: 6366.38}

# Functional forms used by the Rainbow fit (see `fit_rainbow` and
# https://github.com/light-curve/light-curve-python): a sigmoid for the
# blackbody temperature evolution, and a Bazin function for the bolometric
# flux evolution.
temperature = "sigmoid"
bolometric = "bazin"

# Minimum number of valid photometric points (all bands combined) required
# before attempting feature extraction on a light curve.
min_points_total = 7

# Minimum number of valid photometric points required in *each* band
# (g and r) before attempting feature extraction. Below this, per-band fits
# (Rainbow, salt) are considered unreliable.
min_points_perband = 3

# Minimum time span (in days) required both for a source to be considered
# old enough to be scored (`processor.superluminous_score`) and for its
# light curve to be long enough for feature extraction
# (`slsn_classifier.extract_features`).
min_duration = 20

# Upper bound on the peak absolute magnitude (mag) above which a source is
# considered too faint to genuinely be a superluminous supernova
# (typical SLSNe peak around M ~ -21 mag; this threshold is kept
# conservative). Alerts classified as SLSN candidates but whose faintest
# plausible absolute magnitude (given photo-z uncertainty) is fainter than
# this value have their probability forced to 0, see
# `processor.superluminous_score` and `slsn_classifier.abs_peak`.
not_sl_threshold = -19.75
