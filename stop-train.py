#!/usr/bin/env python3
# kate: space-indent on; indent-width 4; mixedindent off; indent-mode python;
"""
Usage:      python stop-train.py $experiment
Motivation: Calls qdel after finding the jobid for the $experiment.
"""
import sys
from argparse import ArgumentParser
from tools import *
from subprocess import check_call

parser = ArgumentParser()
parser.add_argument('train_setup')
parser.add_argument('--ok_if_no_jobs', action="store_true", help="if not set, will exit with error if no jobs found")
args = parser.parse_args()


train_setup = args.train_setup
jobs = find_train_jobs(train_setup)
if not jobs:
    print("No jobs found.")
    sys.exit(0 if args.ok_if_no_jobs else 1)
print("stopping %s" % jobs_repr(jobs))
job_ids = sorted(set([job["id"] for job in jobs]))
cmd = ["qdel"] + list(map(str, job_ids))
print(" ".join(cmd))
check_call(cmd)


