#!/usr/bin/env python3

import faulthandler
faulthandler.enable()
import better_exchook
better_exchook.install()

from argparse import ArgumentParser
from tools import *

parser = ArgumentParser()
parser.add_argument('--alljobs', action='store_true')
args = parser.parse_args()


num_completed_finished = 0

print("# Train:")
print("completed:")
for _, s in sorted(train_setups.items()):
    if not s["completed"]: continue
    prefix = ""
    if train_setup_finished(s):
        num_completed_finished += 1
        if args.alljobs:
            prefix = "*"
        else:
            continue
    print("  %s%s" % (prefix, train_setup_repr(s)))

if args.alljobs:
    if num_completed_finished > 0:
        print("  (*: finished)")
else:
    if num_completed_finished > 0:
        print("  (%i left out because they are finished. show with --alljobs)" % num_completed_finished)

print("incomplete, running:")
for _, s in sorted(train_setups.items()):
    if s["completed"]: continue
    if not s["jobs"]: continue
    print("  %s" % train_setup_repr(s))

print("incomplete, not running:")
for _, s in sorted(train_setups.items()):
    if s["completed"]: continue
    if s["jobs"]: continue
    # TODO: find out reason
    print("  %s" % train_setup_repr(s))


print("")
print("# Recog:")
recog_setups_eps = {(s["name"], s["epoch"]): s for ss in recog_setups.values() for s in ss.values()}
print("completed:")
for _, s in sorted(recog_setups_eps.items()):
    if not s["completed"]: continue
    if s["jobs"]:
        print("  %s %3i. running(?) %s" % (s["name"], s["epoch"], s["jobs_repr"]))
    else:
        print("  %s %3i" % (s["name"], s["epoch"]))


print("incomplete:")
for _, s in sorted(recog_setups_eps.items()):
    if s["completed"]: continue
    print("  %s %3i: %s" % (s["name"], s["epoch"], s["jobs_repr"]))


# kate: space-indent on; indent-width 4; mixedindent off; indent-mode python;
