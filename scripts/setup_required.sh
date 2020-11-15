#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# Set working directory to directory above this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/../

# Detect cuda version
cudaVersion=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
echo CUDA v$cudaVersion found
# Extract major minor version and remove dot
[[ $cudaVersion =~ ^[0-9]+\.[0-9]+ ]]
cudaVersionMajorMinor="${BASH_REMATCH[0]//./}"

# Install pytorch with detected cuda version
if [[ "$cudaVersionMajorMinor" == "102" ]]
then
	# PyTorch 1.7.0 expects CUDA 10.2 so no addition to the versioning is required
	python -m pip install torch===1.7.0 torchvision===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
else
	# Will add '+cuXXX' to version number e.g. torch===1.7.0+cu110 for cuda 11.0
	python -m pip install torch===1.7.0+cu$cudaVersionMajorMinor torchvision===0.8.1+cu$cudaVersionMajorMinor -f https://download.pytorch.org/whl/torch_stable.html
fi

# OS specific commands
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
	# Using Windows bash with minGW or gygwin
	
	# Numpy needs to be installed before scikit-image can be downloaded
	# Also bug in numpy 1.19.4 on windows so using 1.19.3 (see https://github.com/numpy/numpy/issues/17726)
	python -m pip install numpy==1.19.3
else
	# Using MacOSX or Linux
	
	# Numpy needs to be installed before scikit-image can be downloaded
	python -m pip install numpy
fi

python -m pip install -r requirements.txt