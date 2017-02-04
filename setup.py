#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Setuptools for the pyMTL package.

See:
https://github.com/bibliolytic/pyMTL
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyMTL',
    version='1.0.0',
    description='Multi-task learning models for brain-computer interfacing',
    long_description=long_description,
    url='https://github.com/bibliolytic/pyMTL',
    author='Vinay Jayaram, Karl-Heinz Fiebig',
    author_email='vjayaram@tuebingen.mpg.de, karl-heinz.fiebig@stud.tu-darmstadt.de',
    license='GPL',
    classifiers=[
        'Development Status :: 3 - Alpha',
         # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='bci transfer machine learning  development classification',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=[line.strip() for line in open('requirements.txt', 'r')],
)
