# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

set -ex

mkdir -p checkpoints
cd checkpoints
wget "https://dl.fbaipublicfiles.com/pifuhd/checkpoints/pifuhd.pt" pifuhd.pt
cd ..
