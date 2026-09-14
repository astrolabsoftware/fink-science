#!/bin/bash

pyin=miniconda-py311_25.1.1-2/bin/python3
pyout=miniconda-py313_26.7.1-0/bin/python3

vin=$($pyin -c 'import sklearn;print(sklearn.__version__)')
vout=$($pyout -c 'import sklearn;print(sklearn.__version__)')
echo 'Upgrading models from scikit-learn' ${vin} 'to' ${vout}

MODELFNS='elasticc_rainbow_earlyIa_nometa.pkl model_orphans.pkl'
for MODELFN in ${MODELFNS}; do
    echo '--------------------------------------------------------'
    echo "                   ${MODELFN}                           "
    echo '--------------------------------------------------------'
    ${pyin} 0_serialize.py -modelfn ${MODELFN}
    ${pyout} 1_deserialize.py -modelfn ${MODELFN}
done
