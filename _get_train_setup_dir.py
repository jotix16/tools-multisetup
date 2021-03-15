#!/usr/bin/env python3

import faulthandler
faulthandler.enable()

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../crnn")  # so that we can import Config
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)) + "/../returnn")  # alternative dir
try:
    import returnn  # new-style RETURNN import
except ImportError:
    pass
try:
	from Config import Config
except ImportError:
	print("ImportError. sys.path = %r" % (sys.path,), file=sys.stderr)
	raise

configfile = sys.argv[1]
assert os.path.exists(configfile)

config = Config()
config.load_file(configfile)

model = os.path.splitext(os.path.basename(configfile))[0]
model_dir = config.value("_train_setup_dir", "data-train/" + model)

assert os.path.exists(model_dir)
print(model_dir)
