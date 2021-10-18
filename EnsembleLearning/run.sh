#!/bin/bash
echo "running boosting and bagging algorithms..."
python boostAndBag.py

echo "finding bias, variance, and GSE of trees/bagged trees..."
python bias.py

echo "bonus question: default data..."
python default.py