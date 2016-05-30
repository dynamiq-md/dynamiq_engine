#!/usr/bin/env bash

# Basic idea: get the requirements for OPS and put them in a file. Then have
# conda install them.

OPS_CONDA_RECIPE="https://raw.githubusercontent.com/choderalab/openpathsampling/master/devtools/conda-recipe"
MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

curl ${OPS_CONDA_RECIPE}/meta.yaml | 
    grep "\ \ \ \ \-\ " | 
    sed 's/\ \ \ \ \-\ //' |
    grep -v "openpathsampling" > ${MYDIR}/ops_reqs.txt

cp ${MYDIR}/../conda-recipe/meta.yaml ${MYDIR}/meta.yaml
cat '#!/bin/bash
export ORIG=`pwd`
cd && git clone https://github.com/choderalab/openpathsampling
cd openpathsampling && python setup.py install
cd $ORIG
' > ${MYDIR}/build.sh
