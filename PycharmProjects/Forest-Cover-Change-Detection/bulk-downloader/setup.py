#!/usr/bin/env python

from setuptools import setup
from download_espa_order import __version__

setup(
    # Application name:
    name='download_espa_order',

    # Version number:
    version=__version__,

    # Application author details:
    author='USGS EROS ESPA',

    license=open('UNLICENSE').read(),

    description='Client for downloading ESPA scenes.',
    long_description=open('README.md').read(),

    classifiers = [
        'Programming Language :: Python',
        'Programming Langauge :: Python :: 2.7',
        'Programming Langauge :: Python :: 3.x',
        'Topic :: Scientific/Engineering :: GIS'
    ],

    # Scripts
    # Moves the script to the user's bin directory so that it can be executed.
    # Usage is 'download_espa_order.py' not 'python download_espa_order.py'
    scripts=['download_espa_order.py'],

    # Dependent packages (distributions)
    install_requires=[
        'requests',
        ],

    # Supported Python versions
    python_requires='>=2.7',
)
