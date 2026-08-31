import json
import joblib
import sklearn
import numpy as np
from joblib import load

from sklearn.ensemble import RandomForestClassifier

from sklearn_migrator.classification.random_forest_clf import serialize_random_forest_clf
from sklearn_migrator.dimension.pca import serialize_pca

import argparse

def main():
    """Serialize model into a JSON-compatible dictionary"""
    parser = argparse.ArgumentParser(description="Serialize model into a JSON-compatible dictionary")
    parser.add_argument(
        '-modelfn', type=str, default=None,
        help="Model name"
    )
    args = parser.parse_args(None)

    version_sklearn_in = sklearn.__version__

    model = load('input_{}/'.format(version_sklearn_in) + args.modelfn)

    if args.modelfn.startswith('pca'):
        all_data = serialize_pca(model, version_sklearn_in)
    else:
        all_data = serialize_random_forest_clf(model, version_sklearn_in)

    def convert(o):
        if isinstance(o, (np.integer, np.int64)):
            return int(o)
        elif isinstance(o, (np.floating, np.float64)):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, NoneType):
            return 0
        else:
            raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    with open(args.modelfn.split('.')[0] + '.json', "w") as f:
        json.dump(all_data, f, default=convert)

if __name__ == "__main__":
    main()
