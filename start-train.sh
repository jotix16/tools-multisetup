#!/bin/bash
# Usage: ./start-train.sh $experiment
set -e

cd data-train/$1
# submit with guard 3 (3 tries if it fails)
qint.py train.q.sh -g 3
