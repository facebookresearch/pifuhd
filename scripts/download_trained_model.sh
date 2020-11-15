#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# Set working directory to directory above this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/../

set -ex

mkdir -p checkpoints
cd checkpoints
wget "https://dl.fbaipublicfiles.com/pifuhd/checkpoints/pifuhd.pt" pifuhd.pt
cd ..
