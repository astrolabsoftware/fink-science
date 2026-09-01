#!/bin/bash

pyin=miniconda-py39_25.9.1-3/bin/python3
pyout=miniconda-py313_26.7.1-0/bin/python3

vin=$($pyin -c 'import sklearn;print(sklearn.__version__)')
vout=$($pyout -c 'import sklearn;print(sklearn.__version__)')
echo 'Upgrading models from scikit-learn' ${vin} 'to' ${vout}

MODELFNS='default-model_sigmoid.obj partial.pkl model_20241122_wlimits.pkl rf-1.0.2.obj pca-1.0.2.obj'
for MODELFN in ${MODELFNS}; do
    echo '--------------------------------------------------------'
    echo "                   ${MODELFN}                           "
    echo '--------------------------------------------------------'
    miniconda-py39_25.9.1-3/bin/python3 0_serialize.py -modelfn ${MODELFN}
    miniconda-py313_26.7.1-0/bin/python3 1_deserialize.py -modelfn ${MODELFN}
done
