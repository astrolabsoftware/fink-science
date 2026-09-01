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
import os
import xgboost
import joblib
import pickle

import argparse


def main():
    """Deserialize model into a JSON-compatible dictionary"""
    parser = argparse.ArgumentParser(
        description="Serialize model into a JSON-compatible dictionary"
    )
    parser.add_argument("-modelfn", type=str, default=None, help="Model name")
    args = parser.parse_args(None)

    version_xgboost_out = xgboost.__version__

    model = xgboost.XGBClassifier()
    model.load_model(args.modelfn.split(".")[0] + ".json")

    with open("model_attrs.pkl", "rb") as f:
        attributes_to_keep = pickle.load(f)

    for attr, val in attributes_to_keep.items():
        setattr(model, attr, val)

    outfolder = "xgboost_output_{}".format(version_xgboost_out)
    os.makedirs(outfolder, exist_ok=True)

    joblib.dump(model, "{}/{}".format(outfolder, args.modelfn))
    # model.save_model('{}/{}'.format(outfolder, args.modelfn.split('.')[0] + '.json'))

    print(model)


if __name__ == "__main__":
    main()
