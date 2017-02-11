#!/usr/bin/env python
from setuptools import find_packages, setup

readme = open('README.md').read()

VERSION = '0.1.0'

# same as ./requirements.txt
requirements = [
    'librosa',
    'six',
    'torch',
    'pysndfx'
]

setup(
    # Metadata
    name='torchaudio',
    version=VERSION,
    author='PyTorch Core Team, Sean Naren, Will Frey',
    author_email=('sean.narenthiran@digitalreasoning.com, '
                  'will.frey@digitalreasoning.com'),
    url='https://github.com/pytorch/audio',
    description='audio utilities and datasets for torch deep learning',
    long_description=readme,
    license='BSD',

    # Package info
    packages=find_packages(exclude=('test',)),

    zip_safe=True,
    install_requires=requirements,
)
