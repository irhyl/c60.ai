#!/usr/bin/env python
"""
This is a minimal setup.py that defers to pyproject.toml for configuration.
It's maintained for compatibility with tools that don't support PEP 517/518.
"""

import setuptools

if __name__ == "__main__":
    setuptools.setup()