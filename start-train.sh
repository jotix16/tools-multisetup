#!/bin/bash

set -e

cd data-train/$1
qint.py train.q.sh -g 3
