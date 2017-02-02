#!/usr/bin/env python
from setuptools import setup, find_packages

readme = open('README.md').read()

VERSION = '0.1.0'

# same as ./requirements.txt
requirements = [
    'librosa',
    'torch',
]

setup(
    # Metadata
    name='torchaudio',
    version=VERSION,
    author='PyTorch core devs and Sean Naren',
    author_email='sean.narenthiran@digitalreasoning.com',
    url='https://github.com/pytorch/audio',
    description='Audio utilities and datasets for torch deep learning',
    long_description=readme,
    license='BSD',

    # Package info
    packages=find_packages(exclude=('test',)),

    zip_safe=True,
    install_requires=requirements,
)
