import os
from fink_science import __file__
curdir = os.path.dirname(os.path.abspath(__file__))

classifier_path = curdir + "/data/models/superluminous_classifier.joblib"
band_wave_aa = {1: 4770.0, 2: 6231.0, 3: 7625.0}
temperature='sigmoid'
bolometric='bazin'
min_points_total = 7
min_points_perband = 3