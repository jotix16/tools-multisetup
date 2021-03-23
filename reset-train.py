#!/usr/bin/env python3
"""
Uses:       stop-train.py
Usage:      python reset-train.py $experiment
Motivation: Stops training for $experiment, deletes data-train/$experiment
            and removes the corresponding files in scores/* and logs-archive/*.
"""
from lib.utils import shellcmd
from lib import ui
from tools import *
import os
import sys
import re
from glob import glob
from argparse import ArgumentParser
import better_exchook


better_exchook.install()
parser = ArgumentParser()
parser.add_argument('train_setup')
parser.add_argument('--ignore_non_existing_train_dir', action="store_true")
parser.add_argument('--all_confirmed', action="store_true", help="will not ask for confirmation")
args = parser.parse_args()

if args.all_confirmed:
    print("All confirmed.")
    ui.AllConfirmed = args.all_confirmed

print("Base dir:", os.path.normpath(base_dir))
os.chdir(os.path.normpath(base_dir))
assert os.path.exists("./stop-train.py")

train_setup = args.train_setup
print("** Resetting all of train setup: %s" % train_setup)
train_setup_dir = "data-train/%s" % (train_setup,)
if not os.path.isdir(train_setup_dir):
    print("Train-dir does not exist anymore: %s" % train_setup_dir)
    if args.ignore_non_existing_train_dir:
        print("Proceeding anyway...")
    else:
        sys.exit(1)

else:
    print("* Stopping any running training...")
    shellcmd(["./stop-train.py", train_setup, "--ok_if_no_jobs"])

    print("* Deleting train setup dir...")
    shellcmd(["rm", "-rf", train_setup_dir])

print("* Deleting scores files...")
scores_files = ["scores/%s.train.info.txt" % train_setup, "scores/%s.recog.%ss.txt" % (train_setup, Settings.recog_metric_name)]
scores_files = [f for f in scores_files if os.path.exists(f)]
if scores_files:
    shellcmd(["rm"] + scores_files)
else:
    print("No files in scores/ for setup.")

print("* Collecting logs-archive dirs...")
ds = []
for f in os.listdir("logs-archive"):
    if not f.endswith(".recog"):
        continue
    f2 = f[:-len(".recog")]
    f3, ext = os.path.splitext(f2)
    if not re.match(r"^\.[0-9]+$", ext):
        continue
    if f3 != train_setup:
        continue
    ds += ["logs-archive/%s" % f]
if ds:
    shellcmd(["rm", "-rf"] + ds)
else:
    print("No dirs in logs-archive/ for setup.")

print("* Done.")
