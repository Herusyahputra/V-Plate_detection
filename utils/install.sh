#!/bin/bash

echo -e ">>>Install all dependencies library to run this project<<<<"
sudo apt update -y
sudo apt-get install -y python3-venv
sudo apt-get install build-essential
sudo apt install libexiv2-dev
sudo apt install python3-dev
sudo apt-get install qttools5-dev-tools
sudo apt-get install qttools5-dev
echo -e "1. Create Virtual environment"
virtualenv --python=/usr/bin/python3.6 venv
source ./venv/bin/activate
pip install pip --upgrade
echo -e "2. Install the requirements library"
pip install Pillow==8.1.0 \
	PyQt5==5.15.0 \
	pybind11==2.6.2 \
	opencv-python==4.2.0.32 \
	moildev==2.6.0\
	pyyaml \
	imutils \
	matplotlib \
	opencv-contrib-python==3.4.2.29 \
	sympy
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) utils/exif.cpp -o utils/exif$(python3-config --extension-suffix) `pkg-config --cflags --libs python3` -I/usr/local/include -L/usr/local/lib -lexiv2
echo -e ""
echo -e ">>Good Luck, Haryanto<<"
