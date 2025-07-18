"""
Air Pollution Forecasting with Multivariate LSTM & MLOps Pipeline
IEEE Research Implementation

Title: "Multivariate Time Series Analysis and Batch Normalization for Air Quality Prediction in Long Short-Term Memory Networks"
DOI: 10.1109/INOCON60754.2024.10511808
Conference: 2024 3rd International Conference for Innovation in Technology (INOCON)

Author: Tirumala Manav
GitHub: https://github.com/TirumalaManav/AirPollution-Forecasting-MLOps
Email: Contact via GitHub Issues
Date: July 2024

Description: [Add specific module description here]

License: MIT License
Copyright (c) 2024 Tirumala Manav

This module implements [specific functionality] as part of the complete
air pollution forecasting system with production-ready MLOps pipeline.
"""
import io
import os
from pathlib import Path

from setuptools import find_packages, setup

#Medatdata of the package

NAME = 'prediction_model'
DESCRIPTION = 'Air Pollution Prediction Using LSTM Model'
URL = 'https://github.com/TirumalaManav/AirPollution-Forecasting-MLOps'
EMAIL = 'thirumalamanav123@gmail.com'
AUTHOR = 'Tirumala Manav'
REQUIRES_PYTHON = '>=3.7.0'

pwd = os.path.abspath(os.path.dirname(__file__))

# Get the lsit of packages to be installed
def list_reqs(fname='requirements.txt'):
    with io.open(os.path.join(pwd, fname), encoding='utf-8') as fd:
        return fd.read().splitlines()

try:
    with io.open(os.path.join(pwd, 'README.md'), encoding='utf-8') as fd:
        long_description = '\n' + fd.read()

except FileNotFoundError:
    long_description = DESCRIPTION


# Load the package's __version__.py module as a dictionary.

ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / NAME
about = {}
with open(PACKAGE_DIR / 'VERSION') as f:
    _version = f.read().strip()
    about['__version__'] = _version

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    package_data={'prediction_model': ['VERSION']},
    install_requires=list_reqs(),
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
