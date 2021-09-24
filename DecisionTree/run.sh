#!/bin/bash
echo "running 'cars.py'..."
python cars.py

echo "running 'bank.py'..."
python bank.py

echo "running 'bank.py' and replacing 'unknown's ..."
python bank.py --unknown True