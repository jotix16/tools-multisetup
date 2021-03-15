#!/usr/bin/env python3

"""
lur files, created via sclite.
E.g. from Switchboard setup.
"""

import sys
import os
import re
import gzip
import numpy
from pprint import pprint
from glob import glob
import fnmatch
from argparse import ArgumentParser
# from lib.utils import * #??
from tools import *

r_epoch = re.compile('epoch *([0-9]+)')


def get_best_wer(fn):
  """
  :param str fn:
  :return (wer, epoch), or (None, None)
  :rtype: (int,int)|(None,None)
  """
  ls = []
  for l in open(fn).read().splitlines():
    k, v = l.split(":", 1)
    epoch = r_epoch.match(k).group(1)
    epoch = int(epoch)
    try:
      v = float(v)
    except ValueError as exc:
      print("Warning, get_best_wer %r, line %r: %s" % (fn, l, exc))
      continue
    ls += [(v, epoch)]
  if not ls:
    return None, None
  if Settings.recog_score_lower_is_better:
    return min(ls)
  else:
    return max(ls)


def natural_sort_key(s):
  """
  https://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort

  :param str s:
  :return: sth sortable
  """
  return [int(text) if text.isdigit() else text.lower() for text in re.compile('([0-9]+)').split(s)]


def get_models(regexp_pattern, glob_pattern, sort_by):
  """
  :param str regexp_pattern: if given, will be a regexp pattern for the model name
  :param str glob_pattern: if given, will be a glob pattern for the model name
  :param str sort_by: "wer" or "name"
  :return: list of dict with model/wer/epoch
  :rtype: list[dict[str]]
  """
  best_models = []
  for fn in sorted(glob("scores/*.recog.%ss.txt" % Settings.recog_metric_name)):
    model_name = fn[len("scores/"):-len(".recog.%ss.txt" % Settings.recog_metric_name)]
    if regexp_pattern and not re.fullmatch(regexp_pattern, model_name):
      continue
    if glob_pattern and not fnmatch.fnmatch(model_name, glob_pattern):
      continue
    best_wer, epoch = get_best_wer(fn)
    if epoch is None:
      print("Warning, no scores in %r." % fn)
      continue
    best_models += [{"model": model_name, "wer": best_wer, "epoch": epoch}]
  reverse = False
  if sort_by == "wer":
    key = lambda d: d["wer"]
    reverse = not Settings.recog_score_lower_is_better
  elif sort_by == "name":
    key = lambda d: natural_sort_key(d["model"])
  else:
    assert False, "unexpected sort_by %r" % (sort_by,)
  best_models = sorted(best_models, key=key, reverse=reverse)
  return best_models


def open_lur(fn, expected_sub_corpora, sub_corpora_short):
  """
  :param str fn:
  :param list[str] expected_sub_corpora:
  :param list[str] sub_corpora_short:
  :return: dict short sub corpora -> value
  :rtype: dict[str,float]
  """
  assert len(expected_sub_corpora) == len(sub_corpora_short)
  if fn.endswith(".gz"):
    f = gzip.open(fn, "rt", encoding="utf8")
  else:
    f = open(fn)
  with f:
    state = 0
    for ln in f.read().splitlines():
      # wait for "|\s+|\s+||\s+Corpus .*"
      if state == 0:
        if re.match(r"^\|\s+\|\s+\|\|\s+Corpus\s+.*|$", ln):
          state = 1
        continue
      if state == 1:
        assert re.match(r"^\|-+\+-+.*|$", ln), "unexpected in file %r" % fn
        state = 2
        continue
      assert ln and ln[0] == ln[-1] == "|", "unexpected in file %r" % fn
      ln = ln[1:-1].replace("||", "|")
      parts = [p.strip() for p in ln.split("|")]
      if state == 2:
        assert parts[0] == "SPKR"
        for i, sub_corpus in enumerate(expected_sub_corpora):
          assert parts[i + 1] == sub_corpus, "unexpected in file %r" % fn
        state = 3
        continue
      if state == 3:
        # parse sth like "| Set Sum/Avg | [42993]    17.8 || [21594]     23.7 | [21399]     11.9 ||"
        if parts[0] == "Set Sum/Avg":
          assert len(parts[1:]) >= len(expected_sub_corpora), "unexpected in file %r" % fn
          res = {}
          for sub_corpus, sub_corpus_short, part in zip(expected_sub_corpora, sub_corpora_short, parts[1:]):
            m = re.match(r"\[([0-9]+)\]\s*([0-9.]+)", part)
            assert m, "no match for part %r, unexpected in file %r" % (part, fn)
            _, value = m.groups()
            value = float(value)
            res[sub_corpus_short] = value
          return res
    assert False, "unexpected in file %r, not found, final state %i" % (fn, state)


def _dump_res_latex_format(res):
  """
  :param dict[str,dict[str,float]] res:
  """
  for corpus in res:  # order well defined in Python >=3.6
    for sub_corpus in res[corpus]:
      value = res[corpus][sub_corpus]
      fmt = " & %.1f"
      if value < 1.0:
        fmt = " & %.2f"
      print(fmt % value, end="")
  print(" \\\\")


def _collected_res_reduce(res, f):
  """
  :param dict[str,dict[str,list[float]]] res:
  :param ((list[float])->float) f:
  :rtype: dict[str,dict[str,float]]
  """
  return {
    corpus: {sub_corpus: f(res[corpus][sub_corpus]) for sub_corpus in res[corpus]}
    for corpus in res}


def main():
  assert sys.version_info[:2] >= (3, 6)
  arg_parser = ArgumentParser()
  arg_parser.add_argument("--regexp_pattern")
  arg_parser.add_argument("--glob_pattern")
  arg_parser.add_argument("--sort_by", default="wer", help="wer, name")
  arg_parser.add_argument("--type", help="e.g. swb")
  args = arg_parser.parse_args()
  if not args.type:
    if "/switchboard/" in base_dir:
      args.type = "swb"
  assert args.type, "type unclear, base_dir %r does not help" % base_dir
  models = get_models(regexp_pattern=args.regexp_pattern, glob_pattern=args.glob_pattern, sort_by=args.sort_by)
  print("Found %i models:" % len(models))
  # print Latex friendly
  print("%% || %s || epoch || model ||" % Settings.recog_metric_name.upper())
  collected_res = {}
  for d in models:
    print("%% || %.1f || %i || %s ||" % (d["wer"], d["epoch"], d["model"]), end=" ")
    logs_path = "logs-archive/%s.%03d.recog" % (d["model"], d["epoch"])
    if not os.path.isdir(logs_path):
      print("Warning, no log dir %r." % logs_path)
      continue
    if args.type == "swb":
      prefix = ""
      if os.path.exists("%s/scoring-dev.filt.lur.gz" % logs_path):
        prefix = "scoring-"
      res = {
        "Hub5'00:": open_lur(
          "%s/%sdev.filt.lur.gz" % (logs_path, prefix), ["Overall", "Callhome", "Switchboard"], ["Σ", "CH", "SWB"]),
        "Hub5'01:": open_lur("%s/%shub5e_01.filt.lur.gz" % (logs_path, prefix), ["Overall"], ["Σ"])}
      for corpus in res:
        collected_res.setdefault(corpus, {})
        for sub_corpus in res[corpus]:
          collected_res[corpus].setdefault(sub_corpus, []).append(res[corpus][sub_corpus])
      print(res)
    else:
      assert False, "unexpected type %r" % (args.type,)
    # Latex format
    print("%s, %i" % (d["model"], d["epoch"]), end="")
    _dump_res_latex_format(res)
  # mean and std-dev, in Latex format.
  print("$\min$", end="")
  _dump_res_latex_format(_collected_res_reduce(collected_res, numpy.min))
  print("$\max$", end="")
  _dump_res_latex_format(_collected_res_reduce(collected_res, numpy.max))
  print("$\mu$", end="")
  _dump_res_latex_format(_collected_res_reduce(collected_res, numpy.mean))
  print("$\sigma$", end="")
  _dump_res_latex_format(_collected_res_reduce(collected_res, numpy.std))


if __name__ == "__main__":
  main()
