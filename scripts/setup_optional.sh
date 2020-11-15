#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# Set working directory to directory above this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/../

# Detect python version
pyVersion=$(python -V 2>&1 | grep -Po '(?<=Python )(.+)')
if [[ -z "$pyVersion" ]]
then
	echo "Python not found" 
	exit 1
fi
echo Python v$pyVersion found
# Extract major minor version and remove dot
[[ $pyVersion =~ ^[0-9]+\.[0-9]+ ]]
pyVersionMajorMinor="${BASH_REMATCH[0]//./}"

# Check FFMPEG is installed
ffmpegVersionMessage=$(ffmpeg -version 2>&1)
ffmpegFound=false
if [[ -z "$ffmpegVersionMessage" ]]
then
	ffmpegFound=true
	echo "ffmpeg found"
	echo "$ffmpegVersionMessage"
fi

# OS specific commands
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
	# Using Windows bash with minGW or gygwin
	
	if [ "$ffmpegFound" = false ] ; then
		# Check for ffmpeg exe in script path
		if [ -f "ffmpeg/ffmpeg.exe" ]
		then
			echo "ffmpeg found at pifuhd/ffmpeg/ffmpeg.exe"
		else
			echo "ffmpeg not found. Will download from https://github.com/BtbN/FFmpeg-Builds"
			# Download ffmpeg from BtbN auto build for windows
			ffmpeglink=$(curl -s https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest \
			| grep "browser_download_url.*-win64-gpl.zip" \
			| cut -d : -f 2,3 \
			| tr -d \" )
			echo $ffmpeglink
			curl -o ffmpeg.zip -L $ffmpeglink
			# Extract ffmpeg to easy to read folder 'pifuhd/ffmpeg'
			unzip ffmpeg.zip -d ffmpeg
			rm ffmpeg.zip
			cd ffmpeg
			# Find folder name starting with ffmpeg*...
			ffmpegFolder=$(find . -maxdepth 1 -type d -name 'ffmpeg*' -print -quit)
			# Copy files out of oddly named folder into simple ffmpeg folder
			cd $ffmpegFolder/bin
			cp * ../../
			cd ../..
			rm -r $ffmpegFolder
			cd ..
		fi
	fi

	# Offical OpenGL pip is broken on windows so should use binary from here: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopengl
	# Download the correct wheel based on python version
	# Check python version is one of the supported wheels (35,36,37,38,39)
	if [[ "$pyVersionMajorMinor" -le "39" && "$pyVersionMajorMinor" -ge "35" ]]
	then 
		echo "Installing PyOpenGL..."
		# Wheel files for 3.5,3.6,3.7 have filename with added 'm' e.g. cp35m rather than cp35
		if [[ "$pyVersionMajorMinor" -le "39" && "$pyVersionMajorMinor" -ge "38" ]]
		then
			wheelfile="PyOpenGL-3.1.5-cp${pyVersionMajorMinor}-cp${pyVersionMajorMinor}-win_amd64.whl"
		else
			wheelfile="PyOpenGL-3.1.5-cp${pyVersionMajorMinor}-cp${pyVersionMajorMinor}m-win_amd64.whl"
		fi
		curl -o $wheelfile -L https://download.lfd.uci.edu/pythonlibs/z4tqcw5k/$wheelfile
		# install wheel
		python -m pip install $wheelfile
		# delete wheel file after install
		rm $wheelfile
	else
		echo "Invalid python version found"
		echo "OpenGL python wheels only available for >= 3.5 <= 3.9"
		exit 1
	fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
	# Using Linux (Assumes Ubuntu, Debian, Mint that use 'apt-get install')
	
	# Install ffmpeg if missing
	if [ "$ffmpegFound" = false ] ; then
		echo "Installing ffmpeg..."
		sudo apt-get install ffmpeg
	fi

elif [[ "$OSTYPE" == "darwin"* ]]; then
	# Using MacOSX (Assumes homebrew that uses 'brew install')

	# Install ffmpeg if missing
	if [ "$ffmpegFound" = false ] ; then
		brew install ffmpeg
	fi
fi

python -m pip install -r optional-requirements.txt