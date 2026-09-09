# Migrating models

First copy you model inside folders:

```bash
# 1.0.2 is the version of sklearn used to train models
cp ~/codes/fink-science/fink_science/data/models/default-model_sigmoid.obj input_1.0.2
etc.

# 2.1.4 is the version of xgboost used to train models
cp ~/codes/fink-science/fink_science/data/models/superluminous_classifier.joblib xgboost_input_2.1.4
```

Then install miniconda with targeted Python version (in/out):

```bash
# ZTF
./install_miniconda.sh --version py39_25.9.1-3

# Rubin
./install_miniconda.sh --version py311_25.1.1-2

# Output
./install_miniconda.sh --version py313_26.7.1-0
```

And install dependencies:
```bash
miniconda-py39_25.9.1-3/bin/python3 -m pip install -r requirements_in.txt
miniconda-py313_26.7.1-0/bin/python3 -m pip install -r requirements_out.txt
```

## scikit-learn

### 0.22 to 1.0.2

For microlensing models, we need to first jump from 0.22 to 1.0.2 before any upgrades. Just execute:

```bash
miniconda-py38_22.11.1-1/bin/python3 0_serialize.py -modelfn pca.sav
miniconda-py39_25.9.1-3/bin/python3 1_deserialize.py -modelfn pca.sav

miniconda-py38_22.11.1-1/bin/python3 0_serialize.py -modelfn rf.sav
miniconda-py39_25.9.1-3/bin/python3 1_deserialize.py -modelfn rf.sav

mv output_1.0.2/*.obj input_1.0.2/
```

### 1.0.2 to 1.7.2

Check python version inside the script, and convert using:

```bash
./convert_scikit.sh
```

## xgboost

Check Python version inside the script, and simply execute:

```bash
./convert_xgboost.sh
```

