#!/bin/bash

pytest --cov=mlworks tests/
flake8 mlworks tests docs setup.py --ignore E501