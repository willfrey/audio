#!/usr/bin/env python
from setuptools import find_packages, setup

README = open('README.md').read()

VERSION = '0.1.0'

# same as ./requirements.txt
REQUIREMENTS = [
    'librosa',
    'pysoundfile',
    'six',
    'torch',
]

setup(
    # Metadata
    name='torchaudio',
    version=VERSION,
    author='Will Frey, Sean Narenthiran',
    author_email=('will.frey@digitalreasoning.com, ',
                  ' sean.narenthiran@digitalreasoning.com'),
    url='https://github.com/pytorch/audio',
    description='audio utilities and datasets for torch deep learning',
    long_description=README,
    license='BSD',

    # Package info
    packages=find_packages(exclude=('test',)),

    zip_safe=True,
    install_requires=REQUIREMENTS,
)
