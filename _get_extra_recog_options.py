#!/usr/bin/env python3
"""
USAGE:
$python _get_extra_recog_options.py config-train/setup.config epoch


Called from :
create-recog.sh

"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../crnn")  # so that we can import Config
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)) + "/../returnn")  # alternative dir
sys.path.insert(2, "crnn")  # fallback
sys.path.insert(3, "returnn")  # fallback
try:
    import returnn  # new-style RETURNN import
except ImportError:
    pass
from Config import Config

configfile = sys.argv[1]
assert os.path.exists(configfile)
epoch = int(sys.argv[2])

config = Config()
config.load_file(configfile)

model = os.path.splitext(os.path.basename(configfile))[0] # get name of config without .config at the end

if callable(config.typed_dict.get("_extra_recog_options")):
    extra_recog_option = config.typed_dict["_extra_recog_options"](epoch=epoch)
else:
    extra_recog_option = config.value("_extra_recog_options", "")

print(extra_recog_option)
