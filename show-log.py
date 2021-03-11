#!/usr/bin/env python3

import better_exchook
better_exchook.install()
from argparse import ArgumentParser
import os
import sys
import re
import signal
from subprocess import Popen


show_log_cmd = ["less", "-R", "-S", "+G"]


class SignalForward:
    def __init__(self, proc, signum=signal.SIGINT):
        self.signum = signum
        self.proc = proc

    def signal_handler(self, signum, frame):
        self.proc.send_signal(signum)

    def __enter__(self):
        self.old_handler = signal.getsignal(self.signum)
        signal.signal(self.signum, self.signal_handler)

    def __exit__(self, type, value, tb):
        signal.signal(self.signum, self.old_handler)


def get_latest_qdir_log(path, requested_name=None):
    assert os.path.isdir(path)
    fns = []  # (-job id, fn)
    existing_names = []
    for fn in sorted(os.listdir(path)):
        if fn.startswith(".") or fn.startswith("guard."):
            continue
        m = re.match(r"^(.*)\.o([0-9]+)\.?([0-9]+)?$", fn)
        assert m, "no match: %r" % fn
        name, job_id, sub_id = m.groups()
        if name not in existing_names:
            existing_names.append(name)
        if requested_name and name != requested_name:
            continue
        job_id, sub_id = int(job_id), int(sub_id) if sub_id else None
        fns.append((-job_id, fn))
    if not fns:
        if requested_name and existing_names:
            raise Exception("No log for job name %r found. Existing job names: %r" % (requested_name, existing_names))
        raise Exception("No log files found in: %s" % path)
    fns = sorted(fns)
    fn = fns[0][1]
    fn = "%s/%s" % (path, fn)
    assert os.path.exists(fn)
    return fn


def show_log_from_qdir(path, requested_job_name=None):
    fn = get_latest_qdir_log(path, requested_name=requested_job_name)
    cmd = show_log_cmd + [fn]
    print("$ %s" % " ".join(cmd))
    p = Popen(cmd)
    with SignalForward(p):
        p.wait()


def show_train_log(model, **kwargs):
    path = "data-train/%s/qdir/q.log" % model
    show_log_from_qdir(path, **kwargs)
    

def show_recog_log(model, epoch, **kwargs):
    path = "data-recog/%s.%03d/qdir/q.log" % (model, epoch)
    show_log_from_qdir(path, **kwargs)


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--train", action="store_true")
    arg_parser.add_argument("--recog", action="store_true")
    arg_parser.add_argument("--job_name")
    arg_parser.add_argument("model")
    arg_parser.add_argument("epoch", type=int, nargs='?')
    args = arg_parser.parse_args()

    common_kwargs = {"model": args.model, "requested_job_name": args.job_name}
    train_kwargs = common_kwargs
    recog_kwargs = {"epoch": args.epoch}
    recog_kwargs.update(common_kwargs)

    if args.train:
        assert not args.recog
        assert not args.epoch
        return show_train_log(**train_kwargs)
    
    if args.recog:
        assert args.epoch
        assert not args.train
        return show_recog_log(**recog_kwargs)
        
    if args.epoch:
        return show_recog_log(**recog_kwargs)
    else:
        return show_train_log(**train_kwargs)

    print("I don't know what to do.")
    arg_parser.show_help()
    sys.exit(1)
    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        sys.exit(1)
