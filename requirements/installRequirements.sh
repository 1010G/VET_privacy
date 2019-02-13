#!/bin/bash
# Install requirements for the VET_privacy project
pip3 install -r ./requirementsPIP3.txt
#git clone https://github.com/tensorflow/privacy 			# décommenter cette ligne pour télécharger la dernière version du module privacy
pip3 install -e ./privacy/
