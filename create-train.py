#!/usr/bin/env python3

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

# Just take the base config filename as name for the training.
# This also includes the epoch.
model = args.model
config_file = "config-train/%s.config" % model
assert os.path.exists(config_file)

train_dir = "data-train/%s" % model
print("Create %s" % train_dir)

os.mkdir(train_dir)
os.mkdir(train_dir + "/qdir")
os.mkdir(train_dir + "/log")
os.mkdir(train_dir + "/net-model")

shutil.copy("train.q.sh", train_dir)
with open(train_dir + "/settings.sh", "w") as f:
    f.write("model=%s\n" % model)
    f.write("setup_basedir=%s\n" % mydir)

os.symlink(mydir, train_dir + "/base")

#test -e $mydir/theano-cuda-activate.sh && ln -s $mydir/theano-cuda-activate.sh $train_dir/
for f in [
        "data-train", "tools", "tools-recog",
        "flow", "features", "features.warped",
        "config", "config-train", "dependencies",
        "sprint-executables"]:
    if os.path.exists("%s/%s" % (mydir, f)):
        os.symlink("base/%s" % f, "%s/%s" % (train_dir, f))

# Doesnt work for now. Maybe git-new-workdir + submodule combination is broken?

# Create CRNN branch which we use for the setup.
#cd crnn
#git branch $model -f
#cd ..

#echo "git-new-workdir crnn"
#git-new-workdir crnn $train_dir/crnn $model

# Fall-back:
os.symlink("base/%s" % Settings.returnn_dir_name, "%s/%s" % (train_dir, Settings.returnn_dir_name))

assert os.path.exists("%s/%s/rnn.py" % (train_dir, Settings.returnn_dir_name))
