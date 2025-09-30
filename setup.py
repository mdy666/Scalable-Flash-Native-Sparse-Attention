# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='flash-nsa',
    version=1.0,
    description='Triton-based efficient and scalable Native-Sparse-Attention for hopper gpu',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Duyue Ma',
    author_email='maduyue1031@gmail.com',
    url='https://github.com/mdy666/Scalable-Flash-Native-Sparse-Attention',
    packages=find_packages(),
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    python_requires='>=3.9',
    install_requires=[
        "triton>=3.4",
        "torch"
    ],
    extras_require={
    }
)
