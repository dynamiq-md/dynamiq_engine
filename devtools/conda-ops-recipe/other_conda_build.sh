#!/usr/bin/env bash

# Basic idea: use the OPS conda recipe (modified to check out current
# master) and make this directory into a conda recipe for it. Then we can
# `conda build` this directory to include the pre-requisites.

# directory where the OPS conda recipe lived
OPS_CONDA_RECIPE="https://raw.githubusercontent.com/choderalab/openpathsampling/master/devtools/conda-recipe"
MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# we'll replace the local source with the GitHub URL
GITHUB="https://github.com/choderalab/openpathsampling.git"

curl ${OPS_CONDA_RECIPE}/meta.yaml | 
    sed "s|path\:\ \.\.\/\.\.\/|git_url\:\ $GITHUB|" > ${MYDIR}/meta.yaml
curl ${OPS_CONDA_RECIPE}/build.sh > ${MYDIR}/build.sh
