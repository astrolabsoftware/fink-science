# Copyright 2026 AstroLab Software
# Author: Mousa Shaya
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
"""Check that the pre-trained models used by the science modules load (#413)

Each entry below is a model file that a science module actually loads, with
the loader that module uses and the call site it mirrors. The list follows
the code's load calls rather than the content of `data/models/`, which also
holds superseded artifacts that no module loads anymore.

The CI images set SURVEY (ztf or rubin) and each carries only its survey's
dependencies (e.g. no xgboost in the rubin image), so only that survey's
models are checked. Without SURVEY, all models are checked.

Run with `python fink_science/model_check.py` (part of `run_tests.sh`).
"""

import os
import tempfile
import zipfile

from fink_science.tester import regular_unit_tests

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "models")
SURVEY = os.environ.get("SURVEY")

# (survey, relative path under data/models, loader, call site)
ALL_MODELS = [
    # scikit-learn / xgboost, loaded with joblib
    (
        "ztf",
        "sklearn_1.7.2/default-model_sigmoid-1.7.2.obj",
        "joblib",
        "ztf/random_forest_snia/processor.py",
    ),
    (
        "ztf",
        "sklearn_1.7.2/model_20241122_wlimits-1.7.2.obj",
        "joblib",
        "ztf/random_forest_snia/processor.py",
    ),
    ("ztf", "sklearn_1.7.2/partial-1.7.2.obj", "joblib", "ztf/kilonova/processor.py"),
    ("ztf", "sklearn_1.7.2/rf-1-1.7.2.obj", "joblib", "ztf/microlensing/classifier.py"),
    (
        "ztf",
        "sklearn_1.7.2/pca-1-1.7.2.obj",
        "joblib",
        "ztf/microlensing/classifier.py",
    ),
    ("ztf", "xgboost_3.4.1/NOMAI.joblib", "joblib", "ztf/superluminous/kernel.py"),
    (
        "rubin",
        "sklearn_1.7.2/elasticc_rainbow_earlyIa_nometa-1.7.2.obj",
        "joblib",
        "rubin/random_forest_snia/processor.py",
    ),
    (
        "rubin",
        "sklearn_1.7.2/model_orphans-1.7.2.obj",
        "joblib",
        "rubin/orphans/classifier.py",
    ),
    ("rubin", "SLSN_rainbow_no_MD.joblib", "joblib", "rubin/slsn/kernel.py"),
    # TensorFlow SavedModel
    (
        "rubin",
        "cats_models/cats_small_nometa_serial_219_savedmodel",
        "tfsm",
        "rubin/cats/processor.py",
    ),
]

# SuperNNova (PyTorch): ztf/snn and rubin/snn load
# `snn_models/<model_name>/model.pt` with model_name chosen at runtime, so
# every model.pt under snn_models is checked (the elasticc_* ones are rubin's)
SNN_DIR = "snn_models"
for _root, _, _names in sorted(os.walk(os.path.join(MODELS_DIR, SNN_DIR))):
    if "model.pt" in _names:
        _rel = os.path.relpath(os.path.join(_root, "model.pt"), MODELS_DIR)
        _survey = (
            "rubin" if _rel.startswith(os.path.join(SNN_DIR, "elasticc")) else "ztf"
        )
        ALL_MODELS.append((_survey, _rel, "torch", f"{_survey}/snn/processor.py"))

MODELS = [m[1:] for m in ALL_MODELS if SURVEY is None or m[0] == SURVEY]

# Anomaly detection forests (ztf only): one zip per model name, each holding
# two ONNX forests (g and r bands), as unpacked by
# ztf/anomaly_detection/processor.py
ANOMALY_DIR = "anomaly_detection"


def load_model(relpath: str, loader: str):
    """Load one model with the given loader, and return the loaded object

    Parameters
    ----------
    relpath: str
        Path relative to `fink_science/data/models`
    loader: str
        One of `joblib`, `torch`, `tfsm`

    Returns
    -------
    out: object
        The loaded model

    Examples
    --------
    >>> relpath, loader, _ = MODELS[0]
    >>> clf = load_model(relpath, loader)
    >>> hasattr(clf, "predict_proba")
    True
    """
    path = os.path.join(MODELS_DIR, relpath)
    if loader == "joblib":
        import joblib

        return joblib.load(path)
    if loader == "torch":
        import torch

        return torch.load(path, map_location="cpu", weights_only=False)
    if loader == "tfsm":
        import tensorflow as tf

        return tf.keras.layers.TFSMLayer(path, call_endpoint="serving_default")
    raise ValueError(f"Unknown loader {loader}")


def load_anomaly_models():
    """Unzip every anomaly detection model and open its ONNX forests

    Returns
    -------
    out: dict
        Number of ONNX forests opened, keyed by zip file name
        (empty when SURVEY is rubin)

    Examples
    --------
    >>> out = load_anomaly_models()
    >>> SURVEY == "rubin" or (len(out) > 0 and set(out.values()) == {2})
    True
    """
    if SURVEY == "rubin":
        return {}
    import onnxruntime as rt

    zdir = os.path.join(MODELS_DIR, ANOMALY_DIR)
    out = {}
    for fn in sorted(os.listdir(zdir)):
        if not fn.endswith(".zip"):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(os.path.join(zdir, fn)) as zf:
                zf.extractall(tmp)
            sessions = [
                rt.InferenceSession(os.path.join(root, name))
                for root, _, names in os.walk(tmp)
                for name in names
                if name.endswith(".onnx")
            ]
        out[fn] = len(sessions)
    return out


def check_models(models=None):
    """Load every model, and return the ones that failed

    Parameters
    ----------
    models: list, optional
        List of (relative path, loader, call site). Default is MODELS.

    Returns
    -------
    out: list
        (relative path, call site, error) for each model that failed to load

    Examples
    --------
    >>> check_models()
    []
    """
    failures = []
    for relpath, loader, site in MODELS if models is None else models:
        try:
            load_model(relpath, loader)
        except Exception as e:  # noqa: BLE001, PERF203
            failures.append((relpath, site, f"{type(e).__name__}: {e}"))
    return failures


if __name__ == "__main__":
    """Execute the test suite"""
    globs = globals()
    regular_unit_tests(globs)
