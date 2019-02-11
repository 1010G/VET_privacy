#!/bin/bash
# Install requirements for the VET_privacy project
pip3 install -r ./requirementsPIP3.txt
git clone https://github.com/tensorflow/privacy
pip3 install -e ./privacy/
