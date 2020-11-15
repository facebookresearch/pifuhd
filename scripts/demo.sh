#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# Set working directory to directory above this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/../

python -m apps.simple_test
# python apps/clean_mesh.py -f ./results/pifuhd_final/recon
python -m apps.render_turntable -f ./results/pifuhd_final/recon -ww 512 -hh 512