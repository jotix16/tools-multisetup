
import sys
import os
import re
import typing
import numpy
import h5py
from subprocess import Popen, PIPE
from termcolor import colored
from i6lib.sge import getCurrentJobsMoreInfo
from i6lib.str_ import get_str
from lib.utils import parse_time, repr_time_mins, avoid_wiki_link
from pprint import pprint
# noinspection PyPep8Naming
from lazy_object_proxy import Proxy as lazy  # pip install...
# noinspection PyCompatibility
import faulthandler


faulthandler.enable()

my_dir = os.path.dirname(os.path.realpath(__file__))  # dir of the tools
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class InvalidBaseDir(Exception):
    pass


def check_base_dir(path):
    """
    :param str path:
    """
    for d in ["%s/config-train" % path, "%s/scores" % path]:
        if not os.path.exists(d):
            raise InvalidBaseDir("invalid base dir. did not found: %r" % d)


try:
    check_base_dir(base_dir)
except InvalidBaseDir as exc1:
    # Maybe local dir?
    try:
        check_base_dir(os.getcwd())
    except InvalidBaseDir as exc2:
        print("No base dir found:", exc1, exc2)
        raise exc1
    else:
        base_dir = os.getcwd()
        print("Using cwd base dir:", base_dir)


class Settings:
    """
    These settings can be overriden by ../settings.py. See :func:`load_config_py`.
    """
    base_dir = globals()["base_dir"]
    returnn_dir_name = "crnn"
    recog_score_file = None  # see get_scores_filename
    recog_metric_name = "wer"
    recog_get_score_tool = "get-wer.py"
    recog_score_lower_is_better = True
    default_python = "python3"
    reference_gpu = "GeForce GTX 1080 Ti"

    @classmethod
    def returnn_path(cls):
        """
        :rtype: str
        """
        d = "%s/%s" % (cls.base_dir, cls.returnn_dir_name)
        if not os.path.exists("%s/rnn.py" % d) and os.path.exists(os.path.expanduser("~/Programmierung/crnn")):
            # Simple workaround for local Macbook...
            d = os.path.expanduser("~/Programmierung/crnn")
        return os.path.realpath(d)


def _init():
    settings_file = base_dir + "/settings.py"
    if os.path.exists(settings_file):
        from lib.utils import load_config_py, ObjAsDict
        load_config_py(settings_file, ObjAsDict(Settings))
    assert os.path.exists(Settings.returnn_path())
    sys.path.insert(0, Settings.returnn_path())  # so that we can import Config
    try:
        import returnn  # new-style RETURNN import
    except ImportError:
        pass
    # init log with default verbosity (3)
    from Log import log
    log.initialize()


_init()


from Config import Config


def get_recog_dirs():
    """
    :return: list of (name, epoch, dir)
    :rtype: list[(str,int,str)]
    """
    ds = []
    for d in os.listdir("%s/data-recog" % base_dir):
        if not os.path.isdir("%s/data-recog/%s" % (base_dir, d)):
            continue
        base, ext = os.path.splitext(d)
        if ext[:1] != ".":
            continue
        try:
            ep = int(ext[1:])
        except ValueError:
            continue
        ds += [(base, ep, d)]
    return ds


def check_train_model(model_filename):
    """
    :param str model_filename:
    :rtype: int
    """
    model = h5py.File(model_filename, "r")
    epoch = model.attrs['epoch']
    return epoch


def is_existing_model(setup, epoch):
    """
    :param str setup:
    :param int epoch:
    :rtype: bool
    """
    d = "%s/data-train/%s/net-model" % (base_dir, setup)
    possible_names = ["network.%03i" % epoch, "network.pretrain.%03i" % epoch]
    for name in possible_names:
        if os.path.exists("%s/%s.index" % (d, name)):
            return True
    return False


def find_last_finished_train_model(setup):
    """
    :param str setup:
    :rtype: int|None
    """
    d = "%s/data-train/%s/net-model" % (base_dir, setup)
    assert os.path.isdir(d)
    fns = os.listdir(d)
    epochs = {}  # epoch -> fn
    for fn in fns:
        if fn.endswith(".deleted"):
            # Just treat as if it were there.
            fn = fn[:-len(".deleted")]
        if fn.endswith(".index"):
            # TF file. Just remove that ending.
            fn = fn[:-len(".index")]
        base, ext = os.path.splitext(fn)
        if ext[:1] != ".":
            continue
        if base.endswith(".pretrain"):
            base = base[:-len(".pretrain")]
        if base != "network":
            continue
        epoch = int(ext[1:])
        assert epoch not in epochs
        epochs[epoch] = fn
    for epoch, fn in reversed(sorted(epochs.items())):
        if os.path.exists("%s/%s.deleted" % (d, fn)):
            # Was existing before, already deleted but I guess it was finished.
            return epoch
        if os.path.exists("%s/%s.index" % (d, fn)):
            # TF model. No further check for now.
            assert is_existing_model(setup, epoch)
            return epoch
        try:
            model_epoch = check_train_model("%s/%s" % (d, fn))
        except IOError:  # can happen if not correctly written (not finished writing)
            continue
        assert epoch == model_epoch
        return epoch
    return None


def job_repr(job):
    """
    :param dict[str] job:
    :rtype: str
    """
    s = "%s %i.%s %s" % (job["state"], job["id"], job["tasks"], job["name"])
    if "r" in job["state"]:
        s = colored(s, on_color="on_green")
    return s


def jobs_repr(jobs):
    """
    :param list[dict[str]] jobs:
    :rtype: str
    """
    if not jobs:
        return "no jobs"
    n = 3
    if len(jobs) > n:
        return "jobs [%s, %i more...]" % (", ".join(map(job_repr, jobs[:n])), len(jobs) - n)
    else:
        return "jobs [%s]" % ", ".join(map(job_repr, jobs))


def find_train_jobs(setup):
    """
    :param str setup:
    :rtype: list[dict[str]]
    """
    d = "%s/data-train/%s" % (base_dir, setup)
    assert os.path.isdir(d)
    d = os.path.realpath(d) + "/"
    return [job for job in current_jobs if (os.path.realpath(job["cwd"]) + "/").startswith(d)]


def find_recog_jobs(setup_info):
    """
    :param dict[str] setup_info:
    :rtype: list[dict[str]]
    """
    d = setup_info["dir"]
    d = os.path.realpath(d) + "/"
    return [job for job in current_jobs if (os.path.realpath(job["cwd"]) + "/").startswith(d)]


def find_recogs(setup, info=None):
    """
    :param str setup: setup/model name
    :param dict[str]|None info:
    :return: epoch -> info
    :rtype: dict[int,dict[str]]
    """
    if not info:
        info = {}
    ds = {}
    for recog_setup, epoch, d in recog_dirs:
        if recog_setup != setup:
            continue
        ds[epoch] = {"epoch": epoch, "dir": d, "existing_setup": True}

    wer_scores_file = "%s/scores/%s.recog.%ss.txt" % (base_dir, setup, Settings.recog_metric_name)
    wer_scores = open(wer_scores_file).read().splitlines() if os.path.exists(wer_scores_file) else []
    wer_scores = [re.match('epoch +([0-9]+) *: *(.*)', s).groups() for s in wer_scores]
    wer_scores = {int(ep): float(value) for (ep, value) in wer_scores}  # epoch -> score
    for epoch, score in wer_scores.items():
        if epoch not in ds:
            ds[epoch] = {"epoch": epoch, "existing_setup": False}
        ds[epoch]["score"] = score

    def add_suggest(ep, temp=None):
        if ep in ds:
            return
        if temp is None:
            temp = not info.get("completed", False)
        if not is_existing_model(setup, ep):
            return
        ds[ep] = {"epoch": ep, "existing_setup": False, "suggestion": True, "temporary_suggestion": temp}

    last_epoch = info.get("last_epoch", None)
    num_epochs = info.get("num_epochs", None)
    if wer_scores:
        if not last_epoch or last_epoch < max(wer_scores.keys()):
            info["last_epoch"] = last_epoch = max(wer_scores.keys())
        if num_epochs in wer_scores:
            info["completed"] = True
    if last_epoch:
        if info.get("completed", False):
            add_suggest(last_epoch, temp=False)

        n = 5
        while n <= last_epoch:
            if n * 4 >= num_epochs:
                add_suggest(n, temp=False)
            n *= 2

    # collect suggestions based on dev scores
    train_scores_file = "%s/scores/%s.train.info.txt" % (base_dir, setup)
    train_scores = open(train_scores_file).read().splitlines() if os.path.exists(train_scores_file) else []
    train_scores = [re.match('epoch +([0-9]+) +(.*): *(.*)', s) for s in train_scores]
    train_scores = [m.groups() for m in train_scores if m]
    score_keys = set([m[1] for m in train_scores])
    for score_key in score_keys:
        if not score_key.startswith("dev_"):
            continue
        dev_scores = sorted([(float(value), int(ep)) for (ep, key, value) in train_scores if key == score_key])
        assert dev_scores
        if dev_scores[0][0] == dev_scores[-1][0]:
            # All values are the same (e.g. 0.0), so no information. Just ignore this score_key.
            continue
        if dev_scores[0] == (0.0, 1):
            # Heuristic. Ignore the key if it looks invalid.
            continue
        for _, ep in sorted(dev_scores)[:2]:
            add_suggest(ep)

    return ds


def recog_repr(info):
    """
    :type info: dict[str]
    :rtype: str
    """
    if info["existing_setup"]:
        return "%i*" % info["epoch"]
    elif "score" in info:
        return "%i" % info["epoch"]
    else:
        return colored("%i?" % info["epoch"], "magenta")


def recogs_repr(infos):
    """
    :param dict[int,dict[str]] infos:
    :rtype: str
    """
    if infos:
        return "recogs [%s]" % ", ".join([recog_repr(info) for (ep, info) in sorted(infos.items())])
    else:
        return "no recogs"


def train_setup_repr(info):
    setup = info["name"]
    last_epoch = info["last_epoch"] or 0
    num_epochs = info["num_epochs"]
    jobs = info["jobs_repr"]
    recogs_info = info["recogs"]
    recogs = recogs_repr(recogs_info)
    if not info["existing_setup"]:
        jobs = colored(jobs, "magenta")
    elif not info["jobs"] and not info["completed"]:
        jobs = colored(jobs, "red")  # existing setup and no jobs
    return "%s: epoch %i/%i (%.0f%%), %s, %s" % (
           setup, last_epoch, num_epochs, 100. * last_epoch / num_epochs, jobs, recogs)


def train_setup_finished(info, allow_existing_recog=True):
    """
    normally a train setup is finished = train finished, all recogs finished
    but finished-flag can also be set explicitly

    :param dict[str] info:
    :param bool allow_existing_recog:
    :rtype: bool
    """
    if info.get("finished", False):
        return True
    last_epoch = info["last_epoch"] or 0
    num_epochs = info["num_epochs"]
    if last_epoch < num_epochs:
        return False
    jobs = info["jobs"]
    if jobs:
        return False
    if not info.get("do_recog", True):
        return True
    recogs_info = info["recogs"]
    if not recogs_info:
        return False
    if not max(num_epochs, last_epoch) in recogs_info:
        return False  # should be a suggestion, but anyway
    for recog_info in recogs_info.values():
        if recog_info.get("suggestion", False):
            return False
        if not allow_existing_recog and recog_info["existing_setup"]:
            return False
    return True


def train_setup_get_best_recog(info):
    """
    :param dict[str] info:
    :returns: score, list of epochs with that score; or None, empty list
    :rtype: (float|None, list[int])
    """
    recogs_info = info["recogs"]
    best_epochs = []
    best_score = None
    for epoch, recog_info in sorted(recogs_info.items()):
        if "score" not in recog_info:
            continue
        score = recog_info["score"]
        if best_score is None or best_score == score:
            best_epochs.append(epoch)
            best_score = score
            continue
        is_better = False
        if Settings.recog_score_lower_is_better:
            if score < best_score:
                is_better = True
        else:
            if score > best_score:
                is_better = True
        if is_better:
            best_epochs = [epoch]
            best_score = score
    return best_score, best_epochs


def read_multisetup_info_from_config(configfile):
    """
    :param str configfile:
    :rtype: dict[str,str]
    """
    for line in open(configfile).read().splitlines():
        # find lines like: # multisetup: completed True; reason "this is only a template";
        if line.startswith("# multisetup: ") or line.startswith("// multisetup: "):
            if line[:1] == "#":
                line = line[1:]
            elif line[:2] == "//":
                line = line[2:]
            line = line[len(" multisetup: "):]
            items = line.split(";")
            items_d = {}
            for entry in items:
                entry = entry.strip()
                if not entry:
                    continue
                key, value = entry.split(" ", 1)
                items_d[key] = value.strip()
            return items_d
    return {}


def train_setup_info_via_config(configfile):
    """
    :param str configfile:
    :rtype: dict[str]
    """
    config = Config()
    config.load_file(configfile)
    num_epochs = config.int("num_epochs", 0)
    assert num_epochs > 0, "no num_epochs in %r" % configfile
    d = {"num_epochs": num_epochs}
    other_train_setup_dir = config.value("_train_setup_dir", None)
    if other_train_setup_dir:
        # We interpret this as it is referring to another setup.
        # We don't check that existence here.
        d["train_setup_dir"] = other_train_setup_dir
        d["completed"] = True
        d["finished"] = True
        d["explicit_finished"] = True
        d["finished_reason"] = "referenced different training setup"
    multisetup_info = read_multisetup_info_from_config(configfile)
    d["_multisetup_info"] = multisetup_info
    if multisetup_info.get("finished", "false").lower() == "true":
        d["completed"] = True
        d["finished"] = True
        d["explicit_finished"] = True
        d["finished_mark"] = True
        d["finished_reason"] = multisetup_info.get("finished_reason")
    if multisetup_info.get("do_recog", "true").lower() == "false":
        d["do_recog"] = False
    return d


def get_train_setups():
    """
    :return: name -> info dict
    :rtype: dict[str,dict[str]]
    """
    setups = {}
    for setup in sorted(os.listdir("%s/data-train" % base_dir)):
        d = "%s/data-train/%s" % (base_dir, setup)
        if not os.path.isdir(d):
            continue
        configfile = "%s/config-train/%s.config" % (base_dir, setup)
        if not os.path.exists(configfile):
            continue
        info = {"name": setup, "dir": d, "existing_setup": True}
        info.update(train_setup_info_via_config(configfile))
        last_epoch = find_last_finished_train_model(setup)
        info["last_epoch"] = last_epoch
        jobs = find_train_jobs(setup)
        if jobs:
            info["completed"] = False
            info["finished"] = False
        info["jobs"] = jobs
        info["jobs_repr"] = jobs_repr(jobs)
        info.setdefault("completed", (last_epoch or 0) >= info["num_epochs"])
        info["recogs"] = find_recogs(setup, info)
        setups[setup] = info

    for setup in sorted(os.listdir("%s/config-train" % base_dir)):
        if not setup.endswith(".config"):
            continue
        if setup[:1] == "_":
            continue  # ignore those starting with "_"
        setup = setup[:-len(".config")]
        if setup in setups:
            continue
        configfile = "%s/config-train/%s.config" % (base_dir, setup)
        info = {"name": setup, "existing_setup": False,
                "last_epoch": 0, "completed": False,
                "jobs": [], "jobs_repr": "no setup",
                "recogs": {}
                }
        info.update(train_setup_info_via_config(configfile))
        setups[setup] = info

    return setups


def get_recog_setups():
    """
    :return: name -> epoch -> dict
    :rtype: dict[str,dict[int,dict[str]]]
    """
    setup_bases = set([base for (base, _, _) in recog_dirs])
    setups = {setup: {ep: {"dir": "%s/data-recog/%s" % (base_dir, d),
                           "name": setup,
                           "epoch": ep}
                      for (base, ep, d) in recog_dirs
                      if base == setup}
              for setup in setup_bases}

    for setup in sorted(os.listdir("%s/data-train" % base_dir)):
        d = "%s/data-train/%s" % (base_dir, setup)
        if not os.path.isdir(d):
            continue
        configfile = "%s/config-train/%s.config" % (base_dir, setup)
        if not os.path.exists(configfile):
            continue
        setups.setdefault(setup, {})

    for s in [s for ep_setups in setups.values() for s in ep_setups.values()]:
        completed = os.path.exists("%s/%s" % (s["dir"], scores_filename))
        s["completed"] = completed
        jobs = find_recog_jobs(s)
        s["jobs"] = jobs
        jobs_s = jobs_repr(jobs)
        if not completed and not jobs:
            jobs_s = colored(jobs_s, "red")
        s["jobs_repr"] = jobs_s

    return setups


def get_scores_filename():
    """
    :rtype: str
    """
    if Settings.recog_score_file:
        return Settings.recog_score_file
    recog_settings_file = "%s/recog.settings.sh" % base_dir
    assert os.path.exists(recog_settings_file)
    p = Popen(["bash", "-c", "source recog.settings.sh; set -o posix; set"],
              cwd=base_dir, stdout=PIPE, stderr=PIPE)
    out, err = map(get_str, p.communicate())
    assert p.returncode == 0
    assert not err
    env = {k: v for (k, v) in [l.split("=", 1) for l in out.splitlines() if "=" in l]}
    if "devCorpus" not in env:
        print("$devCorpus not in recog.settings.sh?")
        pprint(env)
    return "scoring-%s%s" % (env["devCorpus"], env.get("score_fn_postfix", ".ci.sys"))


r_gpu = re.compile('average epoch time \'(.*)\'')


def get_gpu_epoch_times(fn):
    """
    :param str fn: scores train file
    :rtype: dict[str,int]
    """
    r = {}
    for l in open(fn).read().splitlines():
        k, v = l.split(":", 1)
        if "average epoch time" not in k:
            continue
        gpu = r_gpu.match(k).group(1)
        time = parse_time(v)
        r[gpu] = time
    return r


def get_base_model_filename_from_network(network):
    """
    :param dict[str] network:
    :rtype: str|None
    """
    assert isinstance(network, dict)
    for hidden in network.values():
        if not isinstance(hidden, dict):
            continue
        if "class" not in hidden:
            continue
        if hidden["class"] == "subnetwork":
            if hidden.get("load", "").startswith("data-train/"):
                return hidden["load"]
            fn = get_base_model_filename_from_network(hidden["subnetwork"])
            if fn:
                return fn
        if hidden["class"] == "chunking_sublayer":
            fn = get_base_model_filename_from_network({"sub": hidden["sublayer"]})
            if fn:
                return fn
    return None


def get_base_model(modelname):
    """
    For a given model, see if this was trained based on another model.
    It's fair to consider this for the total calculation time.

    :param str modelname:
    :return: (base_modelname, epoch) or (None, None)
    :rtype: (str,int)|(None,None)
    """
    configfile = "config-train/%s.config" % modelname
    if not os.path.exists(configfile):
        print("Warning: get_base_model: config not found:", configfile)
        return None, None
    config = Config()
    config.load_file(configfile)
    base_model_filename = config.value("import_model_train_epoch1", "")
    if not base_model_filename and config.has("network") and config.is_typed("network"):
        network = config.typed_value("network")
        base_model_filename = get_base_model_filename_from_network(network)
    if not base_model_filename:
        return None, None
    m = re.search("data-train/(.*)/net-model/network.(pretrain\\.)?([0-9]+)", base_model_filename)
    assert m, "cannot handle: %r" % base_model_filename
    base_modelname, _, epoch = m.groups()
    epoch = int(epoch)
    return base_modelname, epoch


_PossibleTrainKeyPrefixes = {"train_score", "train_error", "dev_score", "dev_error", "devtrain_score", "devtrain_error"}


def get_train_scores(train_scores_file, fixup_keys=True):
    """
    :param str train_scores_file: "scores/*.train.info.txt"
    :param bool fixup_keys:
    :return: dict key -> list[(score, ep)]
    :rtype: dict[str,list[(float,int)]]
    """
    train_scores = {}  # type: typing.Dict[str,typing.List[typing.Tuple[float,int]]]  # key -> list[(score, ep)]
    for l in open(train_scores_file).read().splitlines():
        m = re.match('epoch +([0-9]+) ?(.*): *(.*)', l)
        if not m:
            # print("warning: no match for %r" % l)
            continue
        ep, key, value = m.groups()
        prefix = None
        for prefix_ in _PossibleTrainKeyPrefixes:
            if key.startswith(prefix_):
                assert not prefix  # must be unique
                prefix = prefix_
        if not prefix:
            continue
        ep = int(ep)
        value = float(value)
        train_scores.setdefault(key, []).append((value, ep))

    if fixup_keys:
        get_train_key_fixup_mappings(train_scores, apply_fixup=True)

    return train_scores


def get_train_key_fixup_mappings(train_scores, apply_fixup=False):
    """
    :param dict[str,list[(float,int)]] train_scores: if apply_fixup, will fix inplace
    :param bool apply_fixup: if True, will fix train_scores inplace
    :return: key without postfix (i.e. just prefix, like "dev_error") -> full key (e.g. "dev_error_output")
    :rtype: dict[str,str]
    """
    all_epochs = set()  # type: typing.Set[int]
    keys_by_prefix = {}  # type: typing.Dict[str,typing.Dict[str,typing.Set[int]]]  # prefix -> key without prefix -> set of epochs  # nopep8
    keys_wo_prefix = set()
    for key, scores_by_key in sorted(train_scores.items()):
        prefix = None
        for prefix_ in _PossibleTrainKeyPrefixes:
            if key.startswith(prefix_):
                assert not prefix  # must be unique
                prefix = prefix_
        assert prefix
        key_wo_prefix = key[len(prefix):]
        assert not key_wo_prefix or key_wo_prefix.startswith("_")

        for value, ep in scores_by_key:
            all_epochs.add(ep)
            keys_by_prefix.setdefault(prefix, {}).setdefault(key_wo_prefix, set()).add(ep)
            keys_wo_prefix.add(key_wo_prefix)

    # Now try to fixup the keys.
    # The "problem" is that RETURNN automatically removes the postfix if there is only one single such entry,
    # but then, due to pretraining or so, in later epochs the key-name changes if we get an extra score.
    fixup_mappings = {}  # prefix (key without postfix) -> key with postfix
    for prefix in sorted(_PossibleTrainKeyPrefixes):
        if prefix not in keys_by_prefix:  # not used?
            continue
        if "" not in keys_by_prefix[prefix]:  # removed postfix
            # Kind of arbitrary heuristic: Count how much there is "output" in the key.
            # If there is a unique max, take that as the representative for this prefix.
            others_postfixes = [(postfix.count("output"), postfix) for postfix in keys_by_prefix[prefix]]
            max_count, _ = max(others_postfixes)
            others_postfixes = [postfix for count, postfix in others_postfixes if count == max_count]
            if len(others_postfixes) == 1:
                fixup_mappings[prefix] = prefix + others_postfixes[0]
            continue  # nothing else to do here

        epochs_wo_prefix = keys_by_prefix[prefix][""]
        possible_postfixes = []
        for key_wo_prefix, epochs in sorted(keys_by_prefix[prefix].items()):
            assert isinstance(epochs, set)
            if not epochs.isdisjoint(epochs_wo_prefix):  # not a possible postfix
                continue
            if not epochs.union(epochs_wo_prefix) == all_epochs:  # not a possible postfix
                continue
            for other_prefix in sorted(_PossibleTrainKeyPrefixes):
                if other_prefix == prefix:
                    continue
                if other_prefix not in keys_by_prefix:
                    continue
                if key_wo_prefix not in keys_by_prefix[other_prefix]:
                    continue
                other_epochs = keys_by_prefix[other_prefix][key_wo_prefix]
                if other_epochs.isdisjoint(epochs_wo_prefix):  # there must be an intersection
                    continue
                possible_postfixes.append(key_wo_prefix)
                break
        # print("prefix %r, unnamed, possible postfixes %r" % (prefix, possible_postfixes))
        if len(possible_postfixes) == 1:
            # Found one, non-ambiguously.
            fixup_mappings[prefix] = prefix + possible_postfixes[0]

            if apply_fixup:
                # Merge.
                scores_for_key = train_scores.pop(prefix)  # prefix itself is the key (without postfix)
                key = fixup_mappings[prefix]
                train_scores[key].extend(scores_for_key)

    return fixup_mappings


DefaultKey = "dev_error"


def get_train_scores_by_key(train_scores_file, key=DefaultKey, ignore_key=None,
                            max_epoch=float("inf"), filter_not_reached_max_epoch=False):
    """
    :param str train_scores_file: "scores/*.train.info.txt"
    :param str key: e.g. "dev_error"
    :param str|None ignore_key:
    :param int|float max_epoch:
    :param bool filter_not_reached_max_epoch:
    :return: list[(score, ep)]. we might have epochs multiple times, if the key is ambiguous
    :rtype: list[(float,int)]
    """
    train_scores = get_train_scores(train_scores_file, fixup_keys=False)  # dict key -> list[(score, ep)]
    fixup_mappings = get_train_key_fixup_mappings(train_scores, apply_fixup=True)
    if key in fixup_mappings:
        key = fixup_mappings[key]
    ls = []
    for k, v in sorted(train_scores.items()):
        if ignore_key and ignore_key in k:
            continue
        if not k.startswith(key):
            continue
        if max_epoch is not None and max_epoch != float("inf") and max_epoch:
            if filter_not_reached_max_epoch:
                if max([ep for (score, ep) in v]) < max_epoch:
                    continue
            v = [(score, ep) for (score, ep) in v if ep <= max_epoch]
        ls += v
    return ls


def get_best_train_score(train_scores_file, **kwargs):
    """
    :param str train_scores_file:
    :param kwargs: passed to get_train_scores_by_key
    :return: (score, ep)
    :rtype: (float,int)
    """
    train_scores = get_train_scores_by_key(train_scores_file, **kwargs)
    if not train_scores:
        return None
    return min(train_scores)


class CollectStats:

    def __init__(self, max_epoch=None, filter_model_prefix=None, filter_model_regexp=None,
                 filter_base_model=None, filter_base_model_epoch=None, no_base_model=False,
                 trained=False):
        """
        :param int|None max_epoch:
        :param str|None filter_model_prefix:
        :param str|None filter_model_regexp:
        :param str|None filter_base_model:
        :param int|None filter_base_model_epoch:
        :param bool no_base_model: remove models which are based on another model (import params)
        :param bool trained: remove models which are not trained (e.g. use `_train_setup_dir`)
        """
        self.max_epoch = max_epoch
        self.filter_model_prefix = filter_model_prefix
        self.filter_model_regexp = filter_model_regexp
        self.filter_base_model = filter_base_model
        self.filter_base_model_epoch = filter_base_model_epoch
        self.no_base_model = no_base_model
        self.trained = trained
        if self.no_base_model:
            assert not self.filter_base_model
        if self.filter_base_model:
            assert not self.no_base_model
        self._collect_gpu_time()
        self._collect_best_model_times()
        self._collect_best_model_total_times()
        self._collect_best_models_by_time()

    _r_epoch = re.compile('epoch *([0-9]+)')

    def get_best_wer(self, fn):
        """
        :param str fn: scores recog file
        :return (wer, epoch), or (None, None)
        :rtype: (float,int)|(None,None)
        """
        ls = []
        for l in open(fn).read().splitlines():
            k, v = l.split(":", 1)
            epoch = self._r_epoch.match(k).group(1)
            epoch = int(epoch)
            try:
                v = float(v)
            except ValueError as exc:
                print("Warning, get_best_wer %r, line %r: %s" % (fn, l, exc))
                continue
            if self.max_epoch and epoch > self.max_epoch:
                continue
            ls += [(v, epoch)]
        if not ls:
            return None, None
        if Settings.recog_score_lower_is_better:
            return min(ls)
        else:
            return max(ls)

    def get_wers(self, fn):
        """
        :param str fn: scores recog file
        :rtype: dict[int,float]
        """
        wers = {}  # epoch -> wer
        for l in open(fn).read().splitlines():
            k, v = l.split(":", 1)
            epoch = self._r_epoch.match(k).group(1)
            epoch = int(epoch)
            try:
                v = float(v)
            except ValueError as exc:
                print("Warning, get_wers %r, line %r: %s" % (fn, l, exc))
                continue
            if self.max_epoch and epoch > self.max_epoch:
                continue
            wers[epoch] = v
        return wers

    def _collect_gpu_time(self):
        from glob import glob

        all_gpu_times = {}  # setup -> gpu_times (gpu -> time)
        all_gpus = {}  # gpu -> stat count
        for fn in sorted(glob("scores/*.train.info.txt")):
            modelname = fn[len("scores/"):-len(".train.info.txt")]
            gpu_times = get_gpu_epoch_times(fn)
            all_gpu_times[modelname] = gpu_times
            for gpu in gpu_times.keys():
                all_gpus.setdefault(gpu, 0)
                all_gpus[gpu] += 1
        self.all_gpu_times = all_gpu_times
        self.all_gpus = all_gpus

        reference_gpu = Settings.reference_gpu
        if reference_gpu not in all_gpus:
            print("Warning, reference GPU %s not used in experiments.")
            reference_gpu = sorted([(value, key) for (key, value) in all_gpus.items()])[0][1]
            print("Using this reference GPU instead: %s" % reference_gpu)

        gpu_time_factors = {}  # other_gpu -> {setup -> factor}
        for fn in sorted(glob("scores/*.train.info.txt")):
            modelname = fn[len("scores/"):-len(".train.info.txt")]
            gpu_times = all_gpu_times[modelname]
            if reference_gpu in gpu_times and len(gpu_times) > 1:
                for k, v in sorted(gpu_times.items()):
                    if k == reference_gpu:
                        continue
                    gpu_time_factors.setdefault(k, {})[modelname] = float(v) / gpu_times[reference_gpu]
        self.gpu_time_factors = gpu_time_factors

        gpu_mean_time_factors = {}  # other_gpu -> mean factor
        for gpu, factors in sorted(self.gpu_time_factors.items()):
            factor_mean = numpy.mean(list(factors.values()))
            for model, factor in list(factors.items()):
                if factor > 2 * factor_mean:  # remove huge outliers
                    del factors[model]
            factors = list(factors.values())
            gpu_mean_time_factors[gpu] = float(numpy.mean(factors))
        self.gpu_mean_time_factors = gpu_mean_time_factors

        for gpu in sorted(self.all_gpus.keys()):
            if gpu == Settings.reference_gpu:
                continue
            if gpu not in self.gpu_mean_time_factors:
                print("Warning: no time factor for GPU %s to reference GPU %s" % (gpu, Settings.reference_gpu))
                # Hack/workaround for now, so that the code below does not break.
                self.gpu_mean_time_factors[gpu] = 1.0

    def print_gpu_time_factors(self):
        print("||<-4> time factors gpu / reference_gpu, reference_gpu = %s ||" % (
            avoid_wiki_link(Settings.reference_gpu)))
        print("|| gpu || min,max || mean || var ||")
        for gpu, factors in sorted(self.gpu_time_factors.items()):
            factors = list(factors.values())
            print("|| %s || %.3f,%.3f || %.3f || %.3f ||" % (
                avoid_wiki_link(gpu), min(factors), max(factors), float(numpy.mean(factors)),
                float(numpy.var(factors))))
        print()

    def _collect_best_model_times(self):
        """
        now get best models.
        for each model, only the best WER.
        """
        best_model_times = []  # type: typing.List[typing.Dict[str]]
        from glob import glob

        for fn in sorted(glob("scores/*.recog.%ss.txt" % Settings.recog_metric_name)):
            modelname = fn[len("scores/"):-len(".recog.%ss.txt" % Settings.recog_metric_name)]
            best_wer, epoch = self.get_best_wer(fn)
            if epoch is None:
                print("Warning, no scores in %r." % fn)
                continue
            gpu_times = self.all_gpu_times.get(modelname, None)
            if not gpu_times:
                if self.trained:
                    continue
                # print("Warning: No GPU time for model:", modelname)
                gpu_time = float("inf")
                gpu = "?"
                self.gpu_mean_time_factors["?"] = 1.0
            elif Settings.reference_gpu in gpu_times:
                gpu = Settings.reference_gpu
                gpu_time = gpu_times[gpu]
            else:
                gpu_time, gpu = min([(v, k) for (k, v) in sorted(gpu_times.items())])
                if gpu != Settings.reference_gpu:
                    gpu_time /= self.gpu_mean_time_factors[gpu]
            # GPU time in here is already renormalized for the reference GPU.
            best_model_times += [{"wer": best_wer, "epoch": epoch, "model": modelname, "gpu": gpu, "time": gpu_time}]
        best_model_times = sorted(
            best_model_times, key=lambda d: d["wer"], reverse=not Settings.recog_score_lower_is_better)
        self.best_model_times = best_model_times
        models_dict = {d["model"]: d for d in best_model_times}  # type: typing.Dict[str,typing.Dict[str]]
        self.models_dict = models_dict

        self._set_total_time()

        if self.filter_model_prefix:
            for modelname in sorted(models_dict.keys()):
                if not modelname.startswith(self.filter_model_prefix):
                    self._remove_model(modelname)

        if self.filter_model_regexp:
            for modelname in sorted(models_dict.keys()):
                if not re.match(self.filter_model_regexp, modelname):
                    self._remove_model(modelname)

        if self.filter_base_model:
            for modelname in sorted(models_dict.keys()):
                if modelname == self.filter_base_model:
                    continue
                base_modelname, base_modelepoch = get_base_model(modelname)
                if base_modelname != self.filter_base_model:
                    self._remove_model(modelname)
                    continue
                if self.filter_base_model_epoch and base_modelepoch != self.filter_base_model_epoch:
                    self._remove_model(modelname)
                    continue

        if self.no_base_model:
            for modelname in sorted(models_dict.keys()):
                base_modelname, base_modelepoch = get_base_model(modelname)
                if base_modelname:
                    self._remove_model(modelname)

    def print_best_model_times(self, number=10):
        """
        :param int number:
        """
        print("||<-6> %i best models, times for %s ||" % (number, avoid_wiki_link(Settings.reference_gpu)))
        print(
            "|| %s || epoch || model || time for one epoch || time until this epoch || comment ||" % (
                Settings.recog_metric_name.upper()))
        if number < 0:
            number = len(self.best_model_times)
        for d in self.best_model_times[:number]:
            comment = ""
            time = d["time"]
            if d["gpu"] != Settings.reference_gpu:
                comment = "estimated via %s with %s" % (repr_time_mins(time), avoid_wiki_link(d["gpu"]))
            print("|| %.1f || %i || %s || %s || %s || %s ||" % (
                d["wer"], d["epoch"], d["model"], repr_time_mins(time), repr_time_mins(d["total_time"]), comment))
        print()

    def print_best_models_and_train_scores(self):
        print("||<-4> best models and train scores ||")
        print(
            "|| %s || epoch || best dev/train error, relation || model ||" % (
                Settings.recog_metric_name.upper()))

        for d in self.best_model_times[::-1]:
            scores_filename = "scores/%s.train.info.txt" % d["model"]
            dev_error = train_error = float("inf")
            if os.path.exists(scores_filename):
                dev_errors = get_train_scores_by_key(scores_filename, max_epoch=self.max_epoch, key="dev_error")
                train_errors = get_train_scores_by_key(scores_filename, max_epoch=self.max_epoch, key="devtrain_error")
                if not train_errors:
                    train_errors = get_train_scores_by_key(scores_filename, max_epoch=self.max_epoch, key="train_error")
                if dev_errors and train_errors:
                    dev_error, _ = min(dev_errors)
                    train_error, _ = min(train_errors)
            print("|| %.1f || %i || %.4f/%.4f, %.4f || %s ||" % (
                d["wer"], d["epoch"], dev_error, train_error, dev_error / train_error, d["model"]))
        print()

    def get_total_time(self, modelname, epoch):
        """
        :param str modelname:
        :param int epoch:
        :rtype: float|int
        """
        d = self.models_dict[modelname]
        base_total_time = d["base_total_time"]
        total_time = d["time"] * epoch
        return base_total_time + total_time

    def _set_total_time(self, modelname=None):
        """
        :param str|None modelname:
        """
        models = self.models_dict
        if not modelname:
            for modelname in sorted(models.keys()):
                self._set_total_time(modelname)
            return
        d = models[modelname]
        if d.get("total_time"):
            return  # already calculated?
        total_time = d["time"] * d["epoch"]
        base_modelname, base_modelepoch = get_base_model(modelname)
        if base_modelname:
            if base_modelname in self.models_dict:
                self._set_total_time(base_modelname)
                base_total_time = self.get_total_time(base_modelname, base_modelepoch)
                d["base_total_time"] = base_total_time
                total_time += base_total_time
            else:
                print("Warning: unknown base model %s" % base_modelname)
                d["base_total_time"] = 0
        else:
            d["base_total_time"] = 0
        d["total_time"] = total_time

    def _remove_model(self, modelname):
        """
        :param str modelname:
        """
        self.models_dict.pop(modelname)
        for d in list(self.best_model_times):
            if d["model"] == modelname:
                self.best_model_times.remove(d)

    def _collect_best_model_total_times(self):
        # best model total times (epoch * time per epoch)
        # show model + epoch combinations
        from glob import glob

        model_epoch_wers = []
        for fn in sorted(glob("scores/*.recog.%ss.txt" % Settings.recog_metric_name)):
            modelname = fn[len("scores/"):-len(".recog.%ss.txt" % Settings.recog_metric_name)]
            if modelname not in self.models_dict:  # maybe filtered earlier
                continue
            wers = self.get_wers(fn)
            for epoch, wer in wers.items():
                model_epoch_wers += [(wer, epoch, modelname)]
        model_epoch_wers.sort(reverse=not Settings.recog_score_lower_is_better)

        last_models = {}  # model -> {epoch, wer}
        i = 0
        # filter out some
        while i < len(model_epoch_wers):
            wer, epoch, modelname = model_epoch_wers[i]
            if modelname in last_models:
                if ("%.1f" % wer) == ("%.1f" % last_models[modelname]["wer"]):
                    del model_epoch_wers[i]
                    continue
                if last_models[modelname]["epoch"] < epoch:
                    del model_epoch_wers[i]
                    continue
            last_models[modelname] = {"wer": wer, "epoch": epoch}
            i += 1

        self.model_epoch_wers = model_epoch_wers

    def print_best_model_total_times(self):
        print("||<-6> 15 best model epochs, times for %s ||" % avoid_wiki_link(Settings.reference_gpu))
        print("|| %s || epoch || model || time for one epoch || time until this epoch || comment ||" % (
            Settings.recog_metric_name.upper()))
        for wer, epoch, modelname in self.model_epoch_wers[:15]:
            gpu_times = self.all_gpu_times.get(modelname, None)
            if not gpu_times:
                continue
            gpu_time, gpu = min([(v, k) for (k, v) in gpu_times.items()])
            comment = ""
            if gpu != Settings.reference_gpu:
                comment = "estimated via %s with %s" % (repr_time_mins(gpu_time), avoid_wiki_link(gpu))
            print("|| %s ||" % " || ".join(map(str, [
                "%.1f" % wer, epoch, modelname,
                repr_time_mins(gpu_time), repr_time_mins(self.get_total_time(modelname, epoch)),
                comment])))
        print()

    def _collect_best_models_by_time(self):
        # best models by time
        from glob import glob

        models_by_time = []
        for fn in sorted(glob("scores/*.recog.%ss.txt" % Settings.recog_metric_name)):
            modelname = fn[len("scores/"):-len(".recog.%ss.txt" % Settings.recog_metric_name)]
            if modelname not in self.models_dict:  # maybe filtered earlier
                continue
            wers = self.get_wers(fn)
            gpu_times = self.all_gpu_times.get(modelname, None)
            if not gpu_times:
                continue
            gpu_time, gpu = min([(v, k) for (k, v) in gpu_times.items()])
            comment = ""
            if gpu != Settings.reference_gpu:
                comment = "estimated via %s with %s" % (repr_time_mins(gpu_time), avoid_wiki_link(gpu))
            for epoch, wer in wers.items():
                models_by_time += [
                    (self.get_total_time(modelname, epoch), wer, epoch, modelname, comment)]
        models_by_time.sort()
        self.models_by_time = models_by_time

    def print_best_models_by_time(self):
        last_best_wer = float("inf")
        sign_factor = 1
        if not Settings.recog_score_lower_is_better:
            last_best_wer = float("-inf")
            sign_factor = -1

        models_by_time = list(self.models_by_time)

        i = 0
        # filter out some
        while i < len(models_by_time):
            gpu_time, wer, epoch, modelname, comment = models_by_time[i]
            if wer * sign_factor >= last_best_wer * sign_factor:
                del models_by_time[i]
                continue
            last_best_wer = wer
            i += 1

        print("||<-5> best models by total time on %s ||" % avoid_wiki_link(Settings.reference_gpu))
        print("|| time until epoch || %s || epoch || model || comment ||" % Settings.recog_metric_name.upper())
        for gpu_time, wer, epoch, modelname, comment in models_by_time:
            print("|| %s || %.1f || %i || %s || %s ||" % (repr_time_mins(gpu_time), wer, epoch, modelname, comment))
        print()


current_jobs = lazy(lambda: [job for job in getCurrentJobsMoreInfo() if job["cwd"]])  # type: typing.List[dict]
scores_filename = lazy(get_scores_filename)  # type: str
recog_dirs = lazy(get_recog_dirs)  # type: typing.List[typing.Tuple[str,int,str]]
train_setups = lazy(get_train_setups)  # type: typing.Dict[str,dict]  # name -> dict
recog_setups = lazy(get_recog_setups)  # type: typing.Dict[str,typing.Dict[int,dict]]  # name -> epoch -> dict


def reset_lazy():
    # not really documented:
    # https://github.com/ionelmc/python-lazy-object-proxy/blob/master/src/lazy_object_proxy/cext.c
    current_jobs.__wrapped__ = None
    recog_dirs.__wrapped__ = None
    train_setups.__wrapped__ = None
    recog_setups.__wrapped__ = None


# kate: space-indent on; indent-width 4; mixedindent off; indent-mode python;
