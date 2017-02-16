#!/usr/bin/env python
import codecs
import os.path

from setuptools import find_packages, setup

HERE = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(HERE, 'README.md'), encoding='utf-8') as file:
    LONG_DESCRIPTION = file.read()

VERSION = '0.1.0'

REQUIREMENTS = ['librosa', 'pysoundfile', 'six', 'torch']

setup(
    name='torchaudio',
    version=VERSION,
    description='audio utilities and datasets for torch deep learning',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/willfrey/audio',
    author='Will Frey and Sean Narenthiran',
    author_email=('will.frey@digitalreasoning.com, ',
                  ' sean.narenthiran@digitalreasoning.com'),
    license='BSD',
    packages=find_packages(exclude=['tests']),
    install_requires=REQUIREMENTS,
    zip_safe=True
)
