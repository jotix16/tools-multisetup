#!/usr/bin/env python3
# kate: space-indent on; indent-width 4; mixedindent off; indent-mode python;
"""
collect scores from data-train and data-recog, saves it in scores/$model.train.info.txt and logs-archive/*
"""
from argparse import ArgumentParser
import re
import os
from lib.utils import sysexecOut, sysexec, ShellError, betterRepr
from tools import *
import shutil
from glob import glob
from subprocess import check_call
import better_exchook



# simple wrapper, to eval newbob.data
def EpochData(learningRate, error):
    d = {}
    d["learning_rate"] = learningRate
    d.update(error)
    return d

# nan/inf, for some broken newbob.data
nan = float("nan")
inf = float("inf")

def hackUnicodePy2Literals(s):
    # Not sure why, but sometimes Python2 adds the unicode literal to strings when we store the repr, as for newbob.
    # This function hacks it away.
    # via https://bitbucket.org/vinay.sajip/uprefix/src
    # https://www.python.org/dev/peps/pep-0414/
    # Note: Does not always work. E.g. "'dev_..._bleu': ..." -> "'..._ble': ..."
    return re.sub(r"[uU]([\'\"])", r"\1", s)

def copy_detail(setup_dir, postfix, src_filename):
    trg_filename = "%s/logs-archive/%s.%s/%s" % (base_dir, setup_dir, postfix, os.path.basename(src_filename))
    trg_filename = os.path.normpath(trg_filename)
    if os.path.exists(trg_filename) or os.path.exists(trg_filename + ".gz"):
        return
    if not os.path.isdir(os.path.dirname(trg_filename)):
        os.mkdir(os.path.dirname(trg_filename))
    shutil.copyfile(src_filename, trg_filename)
    if not trg_filename.endswith(".gz"):
        sysexec("gzip", trg_filename)


def parse_time(t):
    ts = t.split(":")
    if len(ts) == 1:
        return int(t)
    elif len(ts) == 2:
        return int(ts[1]) + 60 * int(ts[0])
    elif len(ts) == 3:
        return int(ts[2]) + 60 * int(ts[1]) + 60 * 60 * int(ts[0])
    assert False, t


def repr_time(t):
    hh, mm, ss = t / (60 * 60), (t / 60) % 60, t % 60
    return "%i:%02i:%02i" % (hh, mm, ss)


def collect_train_stats(full_dir, args):
    assert os.path.exists("%s/qdir" % full_dir)
    # Collect relevant log files.
    latest_mtime = 0
    fns = []
    for fn in glob("%s/qdir/q.log/*.o*" % full_dir):
        if os.path.basename(fn).startswith("guard."):
            continue
        mtime = os.path.getmtime(fn)
        latest_mtime = max(mtime, latest_mtime)
        fns += [fn]
    fns = sorted(fns)

    extract_info_file = "%s/qdir/q.log/.extract_info" % full_dir
    if os.path.exists(extract_info_file) and not args.recollect_train_stats:
        cache = eval(open(extract_info_file).read())
        if abs(float(cache["latest_mtime"]) - latest_mtime) < 0.001:
            if args.show_cache_usage or args.verbose:
                print("** use cache for", os.path.relpath(os.path.normpath(full_dir), base_dir))
            return cache["items"]

    # Collect time stats.
    time_stats = {}  # gpu -> times
    for fn in fns:
        try:
            gpu_str_lines = sysexecOut("grep", "--text", "Using gpu device", fn).splitlines()
        except ShellError:
            if args.verbose:
                print("%s: Did not found GPU device info, skip." % fn)
            continue  # not started correctly
        assert gpu_str_lines
        gpus = [re.search("gpu device [0-9]+: ([A-Za-z0-9 ]*)", s).group(1).strip() for s in gpu_str_lines]
        if len(gpus) == 1:
            gpu = gpus[0]
        else:
            gpu = " + ".join(gpus)
        try:
            epoch_score_str_lines = sysexecOut("grep", "--text", "-E", "epoch ..? score", fn).splitlines()
        except ShellError:
            if args.verbose:
                print("%s: Did not found epoch score info, skip." % fn)
            continue  # not started correctly
        times = [parse_time(re.search("elapsed: ([0-9:]+)", l).group(1)) for l in epoch_score_str_lines]
        time_stats.setdefault(gpu, []).extend(times)
    if not time_stats:
        if args.verbose:
            print("No time stats, return.")
        return []

    # Collect num params.
    num_params_all_lines = []
    for fn in fns:
        try:
            num_params_lines = sysexecOut("grep", "--text", "net params #:", fn).splitlines()
        except ShellError:
            if args.verbose:
                print("%s: Did not found num net params, skip." % fn)
            continue  # not started correctly
        num_params_all_lines += num_params_lines
    if not num_params_all_lines:
        return []
    # Take last entry. E.g. with pretraining, the last value should be the right one.
    # Some RETURNN versions returned a float here?
    num_params = float(num_params_all_lines[-1].split()[-1])
    assert int(num_params) == num_params
    num_params = int(num_params)

    res = [("num params", num_params)]
    for gpu, times in sorted(time_stats.items()):
        average_time = sum(times) / len(times)
        res += [("average epoch time '%s'" % gpu, repr_time(average_time))]
    with open(extract_info_file, "w") as f:
        f.write(betterRepr({"latest_mtime": latest_mtime, "items": res}))
    return res


def collect_train(setup, args):
    """
    :param str setup:
    """
    fullfn = "%s/data-train/%s" % (base_dir, setup)
    if not os.path.isdir(fullfn):
        print(fullfn, "not found!")
        return
    newbobfile = "%s/newbob.data" % fullfn
    if not os.path.exists(newbobfile):
        print(newbobfile, "not found!")
        return
    newbob_data = open(newbobfile).read()
    if not newbob_data:
        print(newbobfile, "was empty!")
        return  # disk full while writing... very sad...
    #newbob_data = hackUnicodePy2Literals(newbob_data)
    try:
        data = eval(newbob_data)
    except Exception as e:
        print("%s: Newbob eval exception: %r" % (newbobfile, e))
        raise
    targetfile = "%s/scores/%s.train.info.txt" % (base_dir, setup)
    f = open(targetfile, "w")
    keys = sorted(set(sum([list(info.keys()) for info in data.values()], [])))
    for key in keys:
        for epoch, info in sorted(data.items()):
            if key not in info:
                continue
            f.write("epoch %3i %s: %s\n" % (epoch, key, info[key]))
    for k, v in collect_train_stats(fullfn, args):
        f.write("%s: %s\n" % (k, v))
    f.close()
    print(" ", os.path.basename(targetfile))


def main():
    parser = ArgumentParser()
    parser.add_argument("--train_setup", help="if given, only collect this train setup")
    parser.add_argument('--recollect_train_stats', action='store_true')
    parser.add_argument('--show_cache_usage', action='store_true')
    parser.add_argument('--delete_extracted_recog_dirs', action='store_true')
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.train_setup:
        print("Collect train setup %r" % args.train_setup)
        assert os.path.exists("%s/data-train/%s" % (base_dir, args.train_setup))
        args.verbose = True
        collect_train(setup=args.train_setup, args=args)
        return

    # collect train data
    print("Collect train data...")
    for f in sorted(os.listdir("%s/data-train" % base_dir)):
        collect_train(setup=f, args=args)

    # collect recog data
    print("Collect recog data...")
    setups = {}  # dict[str,dict[str,str]], setup -> epoch -> wer
    finished_recog_setups = []

    for f in sorted(os.listdir("%s/data-recog" % base_dir)):
        fullfn = "%s/data-recog/%s" % (base_dir, f)
        if not os.path.isdir(fullfn): continue
        scorefile = "%s/%s" % (fullfn, scores_filename)
        if not os.path.exists(scorefile): continue
        wer = sysexecOut(Settings.recog_get_score_tool, scorefile).strip()
        setup, ext = os.path.splitext(f)
        epoch = int(ext[1:])
        setups.setdefault(setup, {})[epoch] = wer
        if recog_setups[setup][epoch]["jobs"]: continue  # can happen at the end
        finished_recog_setups += [os.path.relpath(os.path.normpath(fullfn), base_dir)]

        scoring_dirs = sorted(glob("%s/scoring/pass2-*" % fullfn))
        if scoring_dirs:
            score_files = glob("%s/*/*" % scoring_dirs[-1])
            assert score_files
            for fn in score_files:
                copy_detail(f, "recog", fn)
        for fn in glob("%s/log.opt/*.log" % fullfn):
            copy_detail(f, "recog", fn)
        for fn in glob("%s/log/*" % fullfn):
            copy_detail(f, "recog", fn)
        for fn in glob("%s/qdir/q.log/*" % fullfn):
            copy_detail(f, "recog", fn)
        for fn in glob("%s/scoring-*" % fullfn):
            copy_detail(f, "recog", fn)
        if Settings.recog_score_file:
            copy_detail(f, "recog", "%s/%s" % (fullfn, Settings.recog_score_file))


    for setup, wers in sorted(setups.items()):
        targetfile = "%s/scores/%s.recog.%ss.txt" % (base_dir, setup, Settings.recog_metric_name)
        existing_scores = open(targetfile).read().splitlines() if os.path.exists(targetfile) else []
        existing_scores = [re.match('epoch +([0-9]+) *: *(.*)', s).groups() for s in existing_scores]
        existing_scores = {int(key): value for (key, value) in existing_scores}
        scores = existing_scores
        scores.update(wers)
        f = open(targetfile, "w")
        for epoch, wer in sorted(scores.items()):
            f.write("epoch %3i: %s\n" % (epoch, wer))
        f.close()
        print(" ", os.path.basename(targetfile))


    if finished_recog_setups:
        if args.delete_extracted_recog_dirs:
            print("Deleting recog dirs:")
            for fn in finished_recog_setups:
                print(" Delete", fn)
                shutil.rmtree(fn)
            print("Done.")
        else:
            print("Recog dirs can be deleted:")
            for fn in finished_recog_setups:
                print(" ", fn)
            print("  $ rm -rf %s" % " ".join(finished_recog_setups))
    else:
        print("All finished recog dirs already cleaned up.")


    if os.path.exists("%s/extract-recog.py" % base_dir):
        print("Found extract-recog.py, calling it.")
        check_call(["%s/extract-recog.py" % base_dir])


if __name__ == "__main__":
    better_exchook.install()
    main()
