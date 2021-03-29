# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2021] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------


from setuptools import find_packages, setup

REQUIRED_PACKAGES = [
    'numpy',
    'scipy',
    'matplotlib',
]

setup(
    name='pydl',
    packages=find_packages(),
    version='0.1.0',
    description='Python library for Deep Learning using NumPy',
    author='Avinash Ranganath',
    author_email='nash911@gmail.com',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
