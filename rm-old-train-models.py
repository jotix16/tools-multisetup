#!/usr/bin/env python3
# kate: space-indent on; indent-width 4; mixedindent off; indent-mode python;
"""
Collect train setups, which are completed + finished.
Filter out the N best ones - don't delete anything from them.
From the remaining, delete all but the best epoch.
"""

from lib.ui import seriousConfirm
from lib.utils import human_size
from tools import *
import re
import os

Skip_N_best = 20

setups = []  # list[setup]

r_epoch = re.compile('epoch *([0-9]+)')

def get_best_wer(fn):
    ls = []
    for l in open(fn).read().splitlines():
        k, v = l.split(":", 1)
        epoch = r_epoch.match(k).group(1)
        ls += [(float(v), int(epoch))]
    return min(ls)

print("collecting...")
for _, s in sorted(train_setups.items()):
    if not s["completed"]: continue
    if not train_setup_finished(s, allow_existing_recog=False): continue
    wer_fn = "scores/%s.recog.wers.txt" % s["name"]
    if not os.path.exists(wer_fn): continue
    best_wer, best_wer_epoch = get_best_wer(wer_fn)
    s["best_wer"] = best_wer
    s["best_wer_epoch"] = best_wer_epoch
    setups += [s]

setups.sort(key=lambda s: (s["best_wer"], s["best_wer_epoch"]))
setups = setups[Skip_N_best:]
print("skipped %i best setups" % Skip_N_best)

fns_to_delete = []

print("remaining:")
for s in setups:
    print(" ", s["best_wer"], s["best_wer_epoch"], s["name"])
    size = 0
    c = 0
    for ep in range(1, s["num_epochs"] + 1):
        if ep == s["best_wer_epoch"]: continue
        ep_model_fn = "%s/net-model/network.%03d" % (s["dir"], ep)
        if not os.path.exists(ep_model_fn):
            if os.path.exists(ep_model_fn + ".deleted"):
                continue  # ok, we have been here before
            if s.get("explicit_finished", False):
                continue  # can happen
            assert False, "epoch model not found: %i, %s" % (ep, ep_model_fn)
        fns_to_delete += [ep_model_fn]
        size += os.path.getsize(ep_model_fn)
        c += 1
    print("    num epochs to delete:", c)
    print("    size of setup:", human_size(size))

s = 0
for fn in fns_to_delete:
    s += os.path.getsize(fn)

print("total num files: %i" % len(fns_to_delete))
print("total size: %s" % human_size(s))

seriousConfirm("Are you sure you want to delete these files?")

print("Deleting files...")
for fn in fns_to_delete:
    open(fn + ".deleted", "w").close()
    os.remove(fn)

print("Done.")

