#!/bin/bash

pyin=miniconda-py39_25.9.1-3/bin/python3
pyout=miniconda-py313_26.7.1-0/bin/python3

vin=$($pyin -c 'import xgboost;print(xgboost.__version__)')
vout=$($pyout -c 'import xgboost;print(xgboost.__version__)')
echo 'Upgrading models from xgboost' ${vin} 'to' ${vout}

MODELFN=superluminous_classifier.joblib
echo '--------------------------------------------------------'
echo "                   ${MODELFN}                           "
echo '--------------------------------------------------------'
${pyin} 0_xgboost.py -modelfn ${MODELFN}
${pyout} 1_xgboost.py -modelfn ${MODELFN}
