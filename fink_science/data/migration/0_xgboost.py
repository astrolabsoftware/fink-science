import xgboost
import joblib

import argparse

def main():
    """Serialize model into a JSON-compatible dictionary"""
    parser = argparse.ArgumentParser(description="Serialize model into a JSON-compatible dictionary")
    parser.add_argument(
        '-modelfn', type=str, default=None,
        help="Model name"
    )
    args = parser.parse_args(None)

    version_xgboost_in = xgboost.__version__

    try:
        model = xgboost.XGBRFClassifier()
        model = model.load_model('xgboost_input_{}/'.format(version_xgboost_in) + args.modelfn)
    except (UnicodeDecodeError, xgboost.core.XGBoostError) as e:
        # First versions were saved with joblib
        model = joblib.load('xgboost_input_{}/'.format(version_xgboost_in) + args.modelfn)

    model.save_model(args.modelfn.split('.')[0] + '.json')

if __name__ == "__main__":
    main()

