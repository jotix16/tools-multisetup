#!/usr/bin/env python3

import better_exchook
import os
import sys
from argparse import ArgumentParser
from collections import OrderedDict


def main():
	parser = ArgumentParser()
	parser.add_argument('train_setup')
	parser.add_argument('reason')
	args = parser.parse_args()
	print("train setup: %s" % args.train_setup)
	config_filename = "config-train/%s.config" % args.train_setup
	assert os.path.exists(config_filename)
	ls = open(config_filename).read().splitlines()
	assert ls, "empty config"
	if ls[0][:2] != "#!":
		print("Only Python format supported at the moment. First line: %r" % ls[0])
		sys.exit(1)
	# Also see tools.train_setup_info_via_config().
	existing = [i for (i, l) in enumerate(ls) if l.startswith("# multisetup: ") or l.startswith("// multisetup: ")]
	if existing:
		assert len(existing) == 1
		l = ls[existing[0]]
		if l[:1] == "#": l = l[1:]
		elif l[:2] == "//": l = l[2:]
		l = l[len(" multisetup: "):]
		items = l.split(";")
		items_d = OrderedDict()
		for entry in items:
			entry = entry.strip()
			if not entry: continue
			key, value = entry.split(" ", 1)
			items_d[key] = value.strip()
		if "finished" in items_d:
			print("have already finished mark, will not change anything: %s" % l)
			sys.exit(1)
		items_d["finished"] = "True"
		items_d["finished_reason"] = repr(args.reason)
		ls[existing[0]] = "# multisetup: " + " ".join(["%s %s;" % (k, v) for (k, v) in items_d.items()])
	else:
		first_non_comment_line = 0
		while ls[first_non_comment_line][:1] == "#":
			first_non_comment_line += 1
			assert first_non_comment_line < len(ls), "only comments?"
		assert not ";" in args.reason, "cannot encode ';' in finish-reason right now"
		finish_comment_mark = "# multisetup: finished True; finished_reason %r;" % args.reason
		ls.insert(first_non_comment_line, finish_comment_mark)
	f = open(config_filename, "w")
	f.write("".join([l + "\n" for l in ls]))
	f.close()
	print("Finish comment mark was written to config: %r" % args.reason)


if __name__ == "__main__":
	better_exchook.install()
	main()
