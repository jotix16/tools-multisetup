#!/usr/bin/env python3
# kate: space-indent on; indent-width 4; mixedindent off; indent-mode python;


from argparse import ArgumentParser
import re
# from lib.utils import * #??
from tools import *
from i6lib.sge import notOnCluster
import shutil
from glob import glob
from subprocess import check_call, CalledProcessError
from pprint import pprint


dry_run = False
start_recogs_only = False


def extract_scores():
    print("Run extract-scores.py.")
    if dry_run:
        print("Dry-run, not running.")
        return
    check_call(["%s/extract-scores.py" % my_dir, "--delete_extracted_recog_dirs"])
    check_call(["git", "add", "scores"])
    try:
        check_call(["git", "commit", "scores", "-m", "scores"])
    except CalledProcessError:
        pass  # maybe no change


def start_recog(model, epoch):
    if dry_run:
        print("Dry-run, not running.")
        return
    check_call(["%s/start-recog.sh" % base_dir, str(model), str(epoch)])


def create_and_start_recog(model, epoch):
    print("Create and start recog:", model, epoch)
    if dry_run:
        print("Dry-run, not running.")
        return
    check_call(["%s/create-recog.sh" % base_dir, str(model), str(epoch)])
    start_recog(model=model, epoch=epoch)


def create_and_start_relevant_recogs():
    print("Create and start relevant recogs.")
    for model, train_info in sorted(train_setups.items()):
        if train_info.get("finished_mark", False):
            continue
        recogs = train_info["recogs"]  # epoch -> info
        for epoch, recog_info in sorted(recogs.items()):
            if recog_info["existing_setup"]:
                if not start_recogs_only:
                    # Use the start_recogs_only flag explicitly.
                    # Otherwise always ignore existing setups.
                    continue
                recog_setup_info = recog_setups[model][epoch]
                if recog_setup_info["completed"]:
                    continue
                if recog_setup_info["jobs"]:
                    continue
                print("Start existing recog, incomplete, non-running:", model, epoch)
                start_recog(model, epoch)
                continue
            if not recog_info.get("suggestion", False):
                continue
            if recog_info["temporary_suggestion"]:
                continue
            create_and_start_recog(model, epoch)


def all_steps():
    if not start_recogs_only:
        extract_scores()
    create_and_start_relevant_recogs()


def parse_args():
    global dry_run, start_recogs_only, train_setups
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--dry_run", action="store_true")
    arg_parser.add_argument("--start_recogs_only", action="store_true")
    arg_parser.add_argument("setups", nargs="*", help="if not given, check all")
    args = arg_parser.parse_args()
    if args.dry_run:
        dry_run = True
    if notOnCluster:
        if not dry_run:
            print("Run this on the cluster. Enabling dry_run now.")
            dry_run = True
    if args.start_recogs_only:
        start_recogs_only = True
    if args.setups:
        for setup in args.setups:
            assert setup in train_setups, "unknown setup %r. we know only %r" % (setup, sorted(train_setups.keys()))
        train_setups = {setup: train_info for (setup, train_info) in train_setups.items() if setup in args.setups}


if __name__ == "__main__":
    parse_args()
    all_steps()
    print("Done.")
