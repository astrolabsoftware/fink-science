import json
from collections import defaultdict
import os
from pprint import pprint

import numpy as np
import pandas as pd

import processor

lc_columns = processor.column_names

names = (
    "536106200014286",
    "633202300014898",
    "633211400005684",
)
test_datasets = [pd.read_csv(f"tests/{fl}.csv") for fl in names]
test_params = {
    'arr_magpsf': [dt.mag.to_numpy() for dt in test_datasets],
    'arr_jd': [dt.mjd.to_numpy() for dt in test_datasets],
    'arr_sigmapsf': [dt.magerr.to_numpy() for dt in test_datasets],
    'arr_cfid': [np.ones(len(dt.mag)) for dt in test_datasets],
    'arr_oId': [i for i, _ in enumerate(test_datasets)]
}

features = []

for name in names:
    with open(f'tests/{name}.features') as features_file:
        lines = features_file.readlines()[::2]
        features.append({
            split[0]: float(split[1][:-1]) for 
            split in map(lambda ln: ln.split(": "), lines)
        })

result = processor.extract_features_snad_raw(**test_params)
result_features = []
for res in result:
    result_features.append({
        lc_columns[i]: res[0][i + 1] for i in range(len(lc_columns))
    })

errors = defaultdict(dict)
# Tolerated deviance is 5%
for expected, actual, name in zip(features, result_features, names):
    for feature_name, feature_value in actual.items():
        if feature_name not in expected:
            continue
        tolerance = abs((feature_value / 100) * 5)
        expected_value = expected[feature_name]
        diff = abs(expected_value - feature_value) if not (expected_value < 0 and feature_value < 0) else (expected_value)
        if diff < tolerance:
            continue

        errors[name][feature_name] = {"expected": expected_value, "recieved": feature_value}

if errors:
    pprint(errors)
    last_run_path = 'tests/last_run.json'
    if os.path.exists(last_run_path):
        with open('tests/last_run.json', 'r') as last_run:
            last_run_errors = json.load(last_run)  # I've not come up with something to do with it yet
    else:
        last_run_errors = None
        print('NOTE: last run data was not found')
    with open('tests/last_run.json', 'w') as last_run:
        json.dump(errors, last_run, indent=4)
else:
    print("Ok!")
