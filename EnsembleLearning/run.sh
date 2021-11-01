#!/bin/bash
echo "running boosting and bagging algorithms..."
python3 boostAndBag.py

echo "finding bias, variance, and GSE of trees/bagged trees..."
python3 bias.py

echo "bonus question: default data..."
python3 default.py