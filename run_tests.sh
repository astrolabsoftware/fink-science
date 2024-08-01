#!/bin/bash
# Copyright 2019-2022 AstroLab Software
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
## Script to launch the python test suite and measure the coverage.
## Must be launched as fink_test
set -e
message_help="""
Run the test suite of the modules\n\n
Usage:\n
    \t./run_tests.sh [--no-spark] [--single_module]\n\n

Note you need Spark 3.1.3+ installed to fully test the modules.
Otherwise, use the --no-spark argument
"""
export ROOTPATH=`pwd`
# Grab the command line arguments
NO_SPARK=false
while [ "$#" -gt 0 ]; do
  case "$1" in
    --no-spark)
      NO_SPARK=true
      shift 1
      ;;
    --single_module)
      SINGLE_MODULE_PATH=$2
      shift 2
      ;;
    -h)
        echo -e $message_help
        exit
        ;;
  esac
done

# Add coverage_daemon to the pythonpath.
export PYTHONPATH="${SPARK_HOME}/python/test_coverage:$PYTHONPATH"
export COVERAGE_PROCESS_START="${ROOTPATH}/.coveragerc"

# single module testing
if [[ -n "${SINGLE_MODULE_PATH}" ]]; then
  coverage run \
   --source=${ROOTPATH} \
   --rcfile ${ROOTPATH}/.coveragerc ${SINGLE_MODULE_PATH}

  # Combine individual reports in one
  coverage combine

  unset COVERAGE_PROCESS_START

  coverage report -m
  coverage html

  exit 0

fi

# Run the test suite on the utilities
for filename in fink_science/*.py
do
 # Run test suite + coverage
 coverage run \
   --source=${ROOTPATH} \
   --rcfile ${ROOTPATH}/.coveragerc $filename
done

# Run the test suite on the modules
for filename in fink_science/*/*.py
do
 # Skip Spark if needed
 if [[ "$NO_SPARK" = true ]] && [[ ${filename##*/} = 'processor.py' ]] ; then
   echo '[NO SPARK] skipping' $filename
 #elif [[ ${filename#*/} = 'anomaly_detection/processor.py' ]]; then
 #  echo "skip anomaly"
 else
   echo $filename
   # Run test suite + coverage
   coverage run \
     --source=${ROOTPATH} \
     --rcfile ${ROOTPATH}/.coveragerc $filename
 fi
done

# Combine individual reports in one
coverage combine

unset COVERAGE_PROCESS_START

coverage report -m
coverage html
