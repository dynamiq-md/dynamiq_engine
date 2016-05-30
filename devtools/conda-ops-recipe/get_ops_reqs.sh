#!/usr/bin/env bash

# Basic idea: get the requirements for OPS and put them in a file. Then have
# conda install them.

OPS_CONDA_RECIPE="https://raw.githubusercontent.com/choderalab/openpathsampling/master/devtools/conda-recipe"
MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

curl ${OPS_CONDA_RECIPE}/meta.yaml | 
    grep "\ \ \ \ \-\ " | 
    sed 's/\ \ \ \ \-\ //' |
    grep -v "openpathsampling" > ${MYDIR}/ops_reqs.txt
