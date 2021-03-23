#!/usr/bin/env python3
"""
Usage:      python create-train.py $model
Motivation: Create the structure of data-train where the data for different experiments are saved.
            net-mode, log, qlog: keep the information about the training
basedir
├── ...
├── data-train               # Recognition files and links to important parts for epoch=1
│   ├── qlog          		 # dir to collect qlog data
│   ├── log         	     # dir to collect log data
│   ├── net-model            # dir to save the checkpoints
│   ├── train.q.sh           # script to start the training
│   ├── base                 -> basedir
│   ├── data-train           -> basedir/data-train
│   ├── tools                -> basedir/tools
│   ├── tools-recog          -> basedir/tools-recog
│   ├── flow                 -> basedir/flow
│   ├── features             -> basedir/features
│   ├── features.warped      -> basedir/features.warped
│   ├── config               -> basedir/config
│   ├── config-train         -> basedir/config-train
│   ├── dependencies         -> basedir/dependencies
│   ├── sprint-executables   -> basedir/sprint-executables
└──
"""
import faulthandler
faulthandler.enable()
import better_exchook
better_exchook.install()

import argparse
import os
import sys
import shutil
from tools import Settings


argparser = argparse.ArgumentParser()
argparser.add_argument("model")
args = argparser.parse_args()

mydir = Settings.base_dir
print("Base dir:", mydir)
os.chdir(mydir)

# make sure the source of data-train link exists
assert os.path.exists(os.readlink("data-train")), \
    'Path data-train is a broken symlink. You have to call "setup-data-dir.py" to setup the symlink to working dir.'

# Just take the base config filename as name for the training.
# This also includes the epoch.
model = args.model
config_file = "config-train/%s.config" % model
assert os.path.exists(config_file)

train_dir = "data-train/%s" % model
if os.path.exists(train_dir):
    sys.exit("The directory %s exists already. We are stopping the script.." % train_dir)


print("Create %s" % train_dir)
os.mkdir(train_dir)
os.mkdir(train_dir + "/qdir")
os.mkdir(train_dir + "/log")
os.mkdir(train_dir + "/net-model")

# training script
shutil.copy("train.q.sh", train_dir)
# info file
with open(train_dir + "/settings.sh", "w") as f:
    f.write("model=%s\n" % model)
    f.write("setup_basedir=%s\n" % mydir)

# Symlinks
os.symlink(mydir, train_dir + "/base")

for f in [
        "data-train", "tools", "tools-recog",
        "flow", "features", "features.warped",
        "config", "config-train", "dependencies",
        "sprint-executables"]:
    if os.path.exists("%s/%s" % (mydir, f)):
        os.symlink("base/%s" % f, "%s/%s" % (train_dir, f))

# Fall-back:
os.symlink("base/%s" % Settings.returnn_dir_name, "%s/%s" % (train_dir, Settings.returnn_dir_name))

assert os.path.exists("%s/%s/rnn.py" % (train_dir, Settings.returnn_dir_name))
