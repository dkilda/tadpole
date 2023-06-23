#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import setuptools

packages = ["tadpole"]

short_description = "Tadpole provides a differentiable programming framework for tensor calculations."

with open("README.md", "r") as f:
     long_description = f.read()

setuptools.setup(
    name = "Tadpole",
    version = "0.0.1",
    description = short_description,
    long_description = long_description,
    long_description_content_type = "text/markdown",
    license = "Apache 2.0",
    url = "https://github.com/dkilda/tadpole",
    packages = packages, 
    classifiers = [
        "Programming Language :: Python :: 3.8.5",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)

