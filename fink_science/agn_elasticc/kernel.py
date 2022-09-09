import os
from fink_science import __file__

curdir = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER = curdir + "/data/models/AGN_elasticc.pkl"
MINIMUM_POINTS = 4
