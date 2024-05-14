#!/bin/bash
python setup.py bdist_wheel  
pip uninstall dist/*.whl --y
pip install dist/*.whl 