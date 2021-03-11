#!/usr/bin/env python3

import sys, os, re
from argparse import ArgumentParser
from pprint import pprint
from tools import *


# A fast and memory efficient implementation
# by Hjelmqvist, Sten
# https://davejingtian.org/2015/05/02/python-levenshtein-distance-choose-python-package-wisely/
def levenshtein_distance(s, t):
    # degenerate cases
    if s == t:
        return 0
    if len(s) == 0:
        return len(t)
    if len(t) == 0:
        return len(s)

    # create two work vectors of integer distances
    #int[] v0 = new int[t.Length + 1];
    #int[] v1 = new int[t.Length + 1];
    v0 = []
    v1 = []

    # initialize v0 (the previous row of distances)
    # this row is A[0][i]: edit distance for an empty s
    # the distance is just the number of characters to delete from t
    # for (int i = 0; i < v0.Length; i++)
    # v0[i] = i;
    for i in range(len(t)+1):
        v0.append(i)
        v1.append(0)

    for i in range(len(s)):
        # calculate v1 (current row distances) from the previous row v0
        # first element of v1 is A[i+1][0]
        # edit distance is delete (i+1) chars from s to match empty t
        v1[0] = i + 1

        # use formula to fill in the rest of the row
        for j in range(len(t)):
            cost = 0 if s[i] == t[j] else 1
            v1[j + 1] = min(v1[j]+1, v0[j+1]+1, v0[j]+cost)

        # copy v1 (current row) to v0 (previous row) for next iteration
        for j in range(len(t)+1):
            v0[j] = v1[j]

    return v1[len(t)]


# http://rosettacode.org/wiki/Longest_common_subsequence#Python
def longest_common_subsequence(a, b):
    lengths = [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
    # read the substring out from the matrix
    result = []
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x-1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y-1]:
            y -= 1
        else:
            assert a[x-1] == b[y-1]
            result = [a[x-1]] + result
            x -= 1
            y -= 1
    return result


def longest_common_subsequence_distance(a, b):
    """
    like levenshtein_distance but substitutions are not allowed
    """
    return len(a) + len(b) - 2 * len(longest_common_subsequence(a, b))


def match_ref_setup(ref_setup, name, ref_setup_dist=1):
    """
    :param str|None ref_setup: ref setup name
    :param str name: other setup name
    :param int ref_setup_dist:
    """
    if not ref_setup:
        return True
    ps1 = ref_setup.split(".")
    ps2 = name.split(".")
    # levenshtein_distance or longest_common_subsequence_distance
    return levenshtein_distance(ps1, ps2) <= ref_setup_dist


def setup_exists(setup_name):
    """
    :param str setup_name:
    :rtype: bool
    """
    fn = "scores/%s.train.info.txt" % setup_name
    return os.path.exists(fn)


def get_max_epoch(setup_name):
    """
    :param str setup_name:
    :return: ep
    :rtype: int
    """
    fn = "scores/%s.train.info.txt" % setup_name
    train_scores = get_train_scores(fn)  # dict key -> list[(score, ep)]
    return max([max([ep for (score, ep) in v]) for (k, v) in train_scores.items()])


def get_setups_with_best_train_score(prefix=None, ref_setup=None, ref_setup_dist=1, **kwargs):
    """
    :param str|None prefix:
    :param str|None ref_setup:
    :param int ref_setup_dist:
    :param kwargs: passed to get_train_scores_by_key
    :return: list[(score, setup_name, ep)]
    :rtype: list[(float,str,int)]
    """
    ls = []  # list[(score, setup_name, ep)]
    from glob import glob
    for fn in glob("scores/*.train.info.txt"):
        setup_name = re.match("scores/(.+)\\.train\\.info\\.txt", fn).group(1)
        assert setup_name
        if prefix and not setup_name.startswith(prefix):
            continue
        if not match_ref_setup(ref_setup, setup_name, ref_setup_dist=ref_setup_dist):
            continue
        best_score = get_best_train_score(fn, **kwargs)
        if not best_score:
            continue
        score, ep = best_score
        ls += [(score, setup_name, ep)]
    return ls


def get_setups_with_best_score_by_epoch(ref_setup=None, ref_setup_dist=1, **kwargs):
    """
    :param str|None ref_setup:
    :param int ref_setup_dist:
    :param kwargs: passed to get_train_scores_by_key
    :return: list[(score, setup_name, ep)]
    :rtype: list[(float,str,int)]
    """
    from glob import glob
    d = {}  # ep -> list[(score,setup)]
    for fn in glob("scores/*.train.info.txt"):
        setup_name = re.match("scores/(.+)\\.train\\.info\\.txt", fn).group(1)
        assert setup_name
        if not match_ref_setup(ref_setup, setup_name, ref_setup_dist=ref_setup_dist):
            continue
        train_scores = get_train_scores_by_key(fn, **kwargs)  # list[(score,ep)]
        for score, ep in train_scores:
            d.setdefault(ep, []).append((score, setup_name))
    l = []  # list[(float,)]
    # TODO not sure if this makes sense ...


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("train_scores_file", nargs='*')
    arg_parser.add_argument("--key", default=DefaultKey, help="dev_error for frame-error-rate, dev_score for loss score")
    arg_parser.add_argument("--ignore_key", help="e.g. dev_score_aoutput")
    arg_parser.add_argument("--max_epoch", default=None, type=int, help="consider scores only up to this epoch")
    arg_parser.add_argument("--filter_not_reached_max_epoch", action="store_true")
    arg_parser.add_argument("--prefix")
    arg_parser.add_argument("--ref_setup")
    arg_parser.add_argument("--ref_setup_dist", type=int, default=1)
    #arg_parser.add_argument("--best_per_epoch", action="store_true")
    args = arg_parser.parse_args()

    if args.max_epoch is not None and args.max_epoch < 0:
        args.max_epoch = float("inf")
    if args.max_epoch is None and args.ref_setup and setup_exists(args.ref_setup):
        args.max_epoch = get_max_epoch(args.ref_setup)
        print("# Using max_epoch by ref_setup:", args.max_epoch)
    if args.max_epoch is None:
        args.max_epoch = float("inf")

    if args.train_scores_file:
        for train_scores_file in args.train_scores_file:
            print("# best scores found in train info file %r" % train_scores_file)
            assert os.path.exists(train_scores_file)
            train_scores = get_train_scores(train_scores_file)
            for key, scores in sorted(train_scores.items()):
                best_score, ep = min(scores)
                epochs = [ep_ for (_, ep_) in scores]
                print(key, "best:", best_score, "in epoch", ep, "(out of epochs %i-%i)" % (min(epochs), max(epochs)))
            return

    if False:  #args.best_per_epoch:
        print("# best setups per epoch by score (%s):" % args.key)
        setups_with_best_score = get_setups_with_best_score_by_epoch(key=args.key, ignore_key=args.ignore_key)
        pprint(sorted(setups_with_best_score))
        return

    print("# setups sorted by their best score (%s):" % args.key)
    setups_with_best_score = get_setups_with_best_train_score(
        prefix=args.prefix,
        key=args.key, ignore_key=args.ignore_key,
        max_epoch=args.max_epoch, filter_not_reached_max_epoch=args.filter_not_reached_max_epoch,
        ref_setup=args.ref_setup, ref_setup_dist=args.ref_setup_dist)
    pprint(sorted(setups_with_best_score))


if __name__ == "__main__":
    import better_exchook
    better_exchook.install()
    main()
