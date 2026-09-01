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
import json
import joblib
import sklearn


from sklearn_migrator.classification.random_forest_clf import (
    deserialize_random_forest_clf,
)
from sklearn_migrator.classification.gradient_boosting_clf import (
    deserialize_gradient_boosting_clf,
)
from sklearn_migrator.dimension.pca import deserialize_pca

import argparse


def main():
    """Deserialize model into a JSON-compatible dictionary"""
    parser = argparse.ArgumentParser(
        description="Deserialize model into a JSON-compatible dictionary"
    )
    parser.add_argument("-modelfn", type=str, default=None, help="Model name")
    args = parser.parse_args(None)

    version_sklearn_out = sklearn.__version__

    with open(args.modelfn.split(".")[0] + ".json", "r") as f:
        all_data = json.load(f)

    if args.modelfn.startswith("pca"):
        new_model = deserialize_pca(all_data, version_sklearn_out)
    elif args.modelfn.startswith("model_orphans.pkl"):
        new_model = deserialize_gradient_boosting_clf(all_data, version_sklearn_out)
    else:
        new_model = deserialize_random_forest_clf(all_data, version_sklearn_out)

    outfolder = "output_{}".format(version_sklearn_out)
    os.makedirs(outfolder, exist_ok=True)

    joblib.dump(
        new_model,
        "{}/{}-{}.obj".format(
            outfolder, args.modelfn.split(".")[0], version_sklearn_out
        ),
    )

    # Test the model
    print(
        joblib.load(
            "{}/{}-{}.obj".format(
                outfolder, args.modelfn.split(".")[0], version_sklearn_out
            )
        )
    )


if __name__ == "__main__":
    main()
