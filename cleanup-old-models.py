#!/usr/bin/env python3

"""
Note: By default, calling this will not do anything. Use --doit to really do it.

We want to cleanup network models (specifically net-model/network.*.{data*,index,meta}).
For each existing train setup:

Via some heuristic, we will calculate a cutoff score, and all setups better than this will not be touched.

If marked via `multisetup: finished True` comment, normally bad and not interesting.
-> Delete all.
If finished training + all recogs:
-> Delete all but the best (w.r.t. recog score).
If multisetup info specifies `keep_epochs`, it will also keep those.

Note that there is also returnn/tools/cleanup-old-models.py, which does not delete as much.
"""

import argparse
from glob import glob
from lib.utils import *
from tools import *
import sys
import os
import re
import ast


def run(args):
    from subprocess import check_call, CalledProcessError
    print("call:", " ".join(args))
    try:
        check_call(args)
    except CalledProcessError:
        print("Error happened.")
        sys.exit(1)


base_dir = os.path.normpath(base_dir)


def find_models(info):
    """
    Return dict epoch -> list of files.
    """
    d = {}
    model_dir = "%s/data-train/%s/net-model" % (base_dir, info["name"])
    assert os.path.isdir(model_dir)
    fn_pattern = "%s/network.*.index" % model_dir
    for fn in glob(fn_pattern):
        re_fn = "(.*/network\\.(pretrain\\.)?)(\\d+)(\\.(broken|crash_\\d+))?\\.index$"
        m = re.match(re_fn, fn)
        assert m, "regexp %r not matched for %r" % (re_fn, fn)
        epoch = int(m.group(3))
        extra_postfix = m.group(4)
        if extra_postfix:  # broken|crash
            assert extra_postfix == ".broken" or extra_postfix.startswith(".crash_")
            key = (epoch, extra_postfix)
        else:
            key = epoch
        fn_prefix = fn[:-len(".index")]
        assert fn == fn_prefix + ".index"
        data_fn_pattern = fn_prefix + ".data*"
        data_fns = glob(data_fn_pattern)
        assert data_fns, "pattern %r not found" % data_fn_pattern
        fns = [fn, fn_prefix + ".meta"] + data_fns
        for fn_ in fns:
            assert os.path.exists(fn_)
        d[key] = fns
    return d


def models_sort_key(item):
    if isinstance(item, int):
        return item, ""
    assert isinstance(item, tuple)
    if isinstance(item[1], list):  # dict item
        return models_sort_key(item[0]) + item[1:]
    return item


def main():
    arg_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    arg_parser.add_argument("--doit", action="store_true", help="only with this flag, we will delete something")
    args = arg_parser.parse_args()

    relevant_train_setups = {}
    best_recog_scores = []  # list of (score, setup_name, ep)

    for setup_name, info in sorted(train_setups.items()):
        if not info["existing_setup"]:
            continue
        # It's not finished if there are still any suggestions to do recog on.
        if not train_setup_finished(info, allow_existing_recog=False):
            continue
        best_recog_score, best_recog_epochs = train_setup_get_best_recog(info)
        relevant_train_setups[setup_name] = info
        for ep in best_recog_epochs:
            best_recog_scores.append((best_recog_score, setup_name, ep))

    print("Out of the finished setups:")
    best_recog_scores.sort(reverse=not Settings.recog_score_lower_is_better)
    if best_recog_scores:
        print("Number of different available scores: %i" % len(best_recog_scores))
        keep_n_best = 10
        print("First %i scores:" % keep_n_best)
        for score, setup_name, ep in best_recog_scores[:keep_n_best]:
            score_str = "%.1f%% %s" % (score, Settings.recog_metric_name.upper())
            print("  %s, setup %s, epoch %i" % (score_str, setup_name, ep))
        score_str = "%.1f%% %s" % (best_recog_scores[-1][0], Settings.recog_metric_name.upper())
        print("Worst score: %s" % score_str)
        if len(best_recog_scores) < keep_n_best:
            cutoff_score = best_recog_scores[-1][0]
        else:
            cutoff_score = best_recog_scores[keep_n_best][0]
        score_str = "%.1f%% %s" % (cutoff_score, Settings.recog_metric_name.upper())
        print("Keep everything for setups with a score better or equal than %s." % score_str)
    else:
        print("No recog scores found?")
        cutoff_score = None

    total_fns_to_delete = []
    total_file_size = 0

    for setup_name, info in sorted(relevant_train_setups.items()):
        assert isinstance(setup_name, str) and isinstance(info, dict)
        best_recog_score, best_recog_epochs = train_setup_get_best_recog(info)
        below_cutoff = False
        if best_recog_epochs:
            if Settings.recog_score_lower_is_better:
                if best_recog_score <= cutoff_score:
                    below_cutoff = True
            else:
                if best_recog_score >= cutoff_score:
                    below_cutoff = True
        if below_cutoff:
            # print("setup %s with score %.1f below cutoff, skip" % (setup_name, best_recog_score))
            continue
        if best_recog_score is None:
            best_recog_score = float("inf") if Settings.recog_score_lower_is_better else float("-inf")
        # TODO: by default, keep nothing.
        #   have clever logic when to keep. e.g. when this is used by other import (import_model_train_epoch1)...
        # Keep the best epochs.
        # If there are no recogs at all, it means we stopped it, usually because it was bad -> cleanup all.
        keep_epochs = list(best_recog_epochs)
        for epoch in ast.literal_eval(info["_multisetup_info"].get("keep_epochs", "[]")):
            assert isinstance(epoch, int)
            keep_epochs.append(epoch)
        delete_epochs = find_models(info)
        for epoch in keep_epochs:
            if epoch in delete_epochs:
                del delete_epochs[epoch]
        if not delete_epochs:  # already cleaned up
            continue
        print("finished setup:", setup_name)
        if keep_epochs:
            print(
                "  best recog: %.1f%% %s, epochs to keep %r" % (
                    best_recog_score, Settings.recog_metric_name.upper(), keep_epochs))
        else:
            print("  (no recog, model was stopped, probably bad)")
        fns_to_delete = []
        for _, fns in sorted(delete_epochs.items(), key=models_sort_key):
            fns_to_delete.extend(fns)
        file_size = 0
        for fn in fns_to_delete:
            file_size += os.path.getsize(fn)
        print("  delete models:", sorted(delete_epochs.keys(), key=models_sort_key), human_size(file_size))
        total_fns_to_delete.extend(fns_to_delete)
        total_file_size += file_size

    print("Total files to delete: %i num, %s" % (len(total_fns_to_delete), human_size(total_file_size)))
    if args.doit:
        print("Deleting now!")
        for fn in total_fns_to_delete:
            os.remove(fn)
        print("Done.")
    else:
        print("Not deleting now. Use --doit.")


if __name__ == "__main__":
    import better_exchook
    better_exchook.install()
    main()
