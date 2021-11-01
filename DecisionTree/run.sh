#!/bin/bash
echo "running 'cars.py'..."
python3 cars.py

echo "running 'bank.py'..."
python3 bank.py

echo "running 'bank.py' and replacing 'unknown's ..."
python3 bank.py --unknown True