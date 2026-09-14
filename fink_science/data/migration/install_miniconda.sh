#!/bin/bash
# Copyright 2022-2024 AstroLab Software
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

# Install Miniconda
# Usage: ./install_miniconda.sh --version <PYTHON_VERSION>

set -e

PYTHON_VERSION=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --version)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$PYTHON_VERSION" ]]; then
  echo "ERROR: --version parameter is required"
  exit 1
fi

echo "Installing Miniconda ${PYTHON_VERSION} in miniconda-${PYTHON_VERSION}..."

# Download and install Miniconda
wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${PYTHON_VERSION}-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p miniconda-${PYTHON_VERSION}
rm /tmp/miniconda.sh

echo "Miniconda ${PYTHON_VERSION} installed successfully in miniconda-${PYTHON_VERSION}"
