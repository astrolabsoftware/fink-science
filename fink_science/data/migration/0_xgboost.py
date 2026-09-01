# Copyright 2019-2026 AstroLab Software
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
import xgboost
import joblib
import pickle

import argparse


def main():
    """Serialize model into a JSON-compatible dictionary"""
    parser = argparse.ArgumentParser(
        description="Serialize model into a JSON-compatible dictionary"
    )
    parser.add_argument("-modelfn", type=str, default=None, help="Model name")
    args = parser.parse_args(None)

    version_xgboost_in = xgboost.__version__

    try:
        model = xgboost.XGBClassifier()
        model = model.load_model(
            "xgboost_input_{}/".format(version_xgboost_in) + args.modelfn
        )
    except (UnicodeDecodeError, xgboost.core.XGBoostError):
        # First versions were saved with joblib
        model = joblib.load(
            "xgboost_input_{}/".format(version_xgboost_in) + args.modelfn
        )

    # The optimal_threshold attribute (and others) is not part of
    # the XGBoost booster serialization
    # it's a scikit-learn wrapper addition
    # So we dump it and we will reload it later
    attributes_to_keep = {}
    for attr in dir(model):
        if not attr.startswith("_") and attr not in dir(xgboost.XGBClassifier()):
            try:
                attributes_to_keep[attr] = getattr(model, attr)
            except AttributeError:
                pass

    # Save separately
    with open("model_attrs.pkl", "wb") as f:
        pickle.dump(attributes_to_keep, f)

    model.save_model(args.modelfn.split(".")[0] + ".json")


if __name__ == "__main__":
    main()
