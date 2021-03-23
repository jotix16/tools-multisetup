#!/usr/bin/env python3

import sys
import os
from argparse import ArgumentParser
from pprint import pprint
import better_exchook
import tools
from lib import ui
"""
"""

my_dir = tools.my_dir
base_dir = tools.base_dir
reset_train_py_path = my_dir + "/reset-train.py"


def cmd(*args):
    from subprocess import check_call
    print("$ " + " ".join(args))
    check_call(args)


def main():
    assert os.path.exists(reset_train_py_path)

    parser = ArgumentParser()
    parser.add_argument('setups', nargs="*", help="setups which should *not* be removed")
    args = parser.parse_args()

    setups = sorted(tools.train_setups.keys())
    if not args.setups:
        print("Error: currently we except that you keep some setups.")
        print("Available setups:")
        pprint(setups)
        sys.exit(1)

    for except_setup in args.setups:
        setups.remove(except_setup)
    print("Setups to remove:")
    pprint(setups)
    print("Setups to keep:")
    pprint(args.setups)
    ui.confirm("Are you sure?")
    for setup in setups:
        setup_train_cfg_fn = "%s/config-train/%s.config" % (base_dir, setup)
        assert os.path.exists(setup_train_cfg_fn)
        cmd(reset_train_py_path, setup, "--all_confirmed", "--ignore_non_existing_train_dir")
        os.remove(setup_train_cfg_fn)


if __name__ == "__main__":
    better_exchook.install()
    main()

