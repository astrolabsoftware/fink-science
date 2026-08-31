import os
import xgboost
import joblib

import argparse

def main():
    """Deserialize model into a JSON-compatible dictionary"""
    parser = argparse.ArgumentParser(description="Serialize model into a JSON-compatible dictionary")
    parser.add_argument(
        '-modelfn', type=str, default=None,
        help="Model name"
    )
    args = parser.parse_args(None)

    version_xgboost_out = xgboost.__version__

    model = xgboost.XGBRFClassifier()
    model.load_model(args.modelfn.split(".")[0] + ".json")

    outfolder = 'xgboost_output_{}'.format(version_xgboost_out)
    os.makedirs(outfolder, exist_ok=True)

    model.save_model('{}/{}'.format(outfolder, args.modelfn.split('.')[0] + '.json'))

    print(model)

if __name__ == "__main__":
    main()

