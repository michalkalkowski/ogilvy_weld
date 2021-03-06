import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="ogilvy_weld",
    version="0.0.1",
    description="An implementation of the Ogilvy geometrical model for orientations in multipass austenitic steel welds",
    author="Michal K Kalkowski",
    author_email="m.kalkowski@imperial.ac.uk",
    packages=find_packages(exclude=['data', 'references', 'output', 'notebooks']),
    long_description=read('README.md'),
    license='MIT',
)
