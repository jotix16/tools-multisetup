#!/usr/bin/env python3

from __future__ import division

from argparse import ArgumentParser
from tools import *


def plot_runtime_vs_wer_to_file(models_by_time, filename, opts):
    """
    :param list[(int,float,int,str,str)] models_by_time: gpu_time, wer, epoch, modelname, comment. already sorted
    :param str filename:
    :param dict[str] opts:
    """
    print("Plot runtime-vs-wer to file:", filename)
    print("Num experiments (recognitions):", len(models_by_time))
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy
    from subprocess import check_output

    models_by_time = list(models_by_time)
    last_best_wer = float("inf")
    last_best_wer_time = 0
    sign_factor = 1
    threshold_factor = opts.get("threshold", 3.)
    time_threshold_factor = opts.get("time_threshold", 1.1)
    if not Settings.recog_score_lower_is_better:  # higher is better, e.g. BLEU
        last_best_wer = -last_best_wer
        sign_factor = -sign_factor
        threshold_factor = 1. / threshold_factor

    i = 0
    while i < len(models_by_time):
        gpu_time, wer, _, _, _ = models_by_time[i]
        if wer * sign_factor < last_best_wer * sign_factor:
            last_best_wer = wer
            last_best_wer_time = gpu_time
        i += 1
    best_wer = last_best_wer
    best_wer_time = last_best_wer_time

    i = 0
    # filter out some outliers (within threshold)
    while i < len(models_by_time):
        gpu_time, wer, _, _, _ = models_by_time[i]
        if gpu_time >= best_wer_time * time_threshold_factor:
            del models_by_time[i]
            continue
        if wer * sign_factor >= best_wer * sign_factor * threshold_factor:
            del models_by_time[i]
            continue
        i += 1

    if False:
        i = 0
        # filter out some, by only become better (within some threshold)
        while i < len(models_by_time):
            _, wer, _, _, _ = models_by_time[i]
            if wer * sign_factor >= last_best_wer * sign_factor * threshold_factor:
                del models_by_time[i]
                continue
            if wer * sign_factor < last_best_wer * sign_factor:
                last_best_wer = wer
            i += 1

    if opts.get("classes"):
        plt.rcParams['figure.figsize'] = 10, 5

    fig, ax = plt.subplots()
    color = None
    classes = None
    cmap = None
    marker = None
    if opts.get("classes"):
        from fnmatch import fnmatch
        classes = []
        for _, _, _, modelname, _ in models_by_time:
            class_idx = len(opts["classes"])
            for i, (_, pattern) in enumerate(opts["classes"]):
                if fnmatch(modelname, pattern):
                    class_idx = i
                    break
            classes.append(class_idx)
        # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        cmap = "jet"
    else:
        config_create_times = []
        cache = {}
        cache_fn = ".cache.time-stats.gittime.txt"
        if os.path.exists(cache_fn):
            cache = eval(open(cache_fn).read())
        for _, _, _, modelname, _ in models_by_time:
            if modelname in cache:
                config_create_times.append(cache[modelname])
                continue
            config_fn = "config-train/%s.config" % modelname
            assert os.path.exists(config_fn)
            t = int(check_output(["git", "log", "--follow", "--format=%at", "--reverse", "--", config_fn]).decode("utf8").splitlines()[0])
            #t = os.stat(config_fn).st_mtime # or ctime?
            cache[modelname] = t
            config_create_times.append(t)
        with open(cache_fn, "w") as f:
            f.write(repr(cache))
        #pprint(config_create_times)
        config_create_times = numpy.array(config_create_times, dtype="float32")
        config_create_times = -config_create_times
        #config_create_times = config_create_times - numpy.mean(config_create_times)
        #config_create_times = config_create_times / numpy.std(config_create_times)
        config_create_times = config_create_times - min(config_create_times)
        config_create_times /= max(config_create_times)
        #pprint(config_create_times)
        color = cm.hot(config_create_times)

    times = numpy.array([item[0] for item in models_by_time]) / 60. / 60. / 24.
    wers = [item[1] for item in models_by_time]
    scatter = ax.scatter(times, wers, c=classes, color=color, cmap=cmap, marker=marker)

    if opts.get("classes"):
        # produce a legend with the unique colors from the scatter
        legend1 = ax.legend(
            *scatter.legend_elements(fmt=matplotlib.ticker.IndexFormatter([name for name, _ in opts["classes"]] + ["Others"])),
            loc="upper right", title="Classes")
        ax.add_artist(legend1)

    ax.set(
        xlabel='Time [days]', ylabel='%s [%%]' % Settings.recog_metric_name.upper(),
        title='Training runtime (%s) vs %s' % (reference_gpu, Settings.recog_metric_name.upper()))
    ax.grid()

    fig.savefig(filename)
    plt.show()


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--max_epoch", default=None, type=int, help="consider scores only up to this epoch")
    arg_parser.add_argument("--filter_model_prefix")
    arg_parser.add_argument("--filter_model_regexp")
    arg_parser.add_argument("--filter_base_model", help="show only models which are based on this model")
    arg_parser.add_argument("--filter_base_model_epoch", type=int)
    arg_parser.add_argument("--no_base_model", action="store_true", help="only models trained from scratch")
    arg_parser.add_argument("--trained", action="store_true", help="only trained models (no pure recog setups)")
    arg_parser.add_argument("--plot_runtime_vs_wer", help="create plot. argument is the output filename (pdf)")
    arg_parser.add_argument("--plot_runtime_vs_wer_opts", default="")
    arg_parser.add_argument("--best_model_setups", type=int, help="only show best model setups. provide number")#
    arg_parser.add_argument("--best_models_and_train_scores", action="store_true")
    args = arg_parser.parse_args()

    os.chdir(base_dir)
    assert os.path.isdir("scores"), "are you in the root of the setup dir?"

    stats = CollectStats(
        max_epoch=args.max_epoch,
        filter_model_prefix=args.filter_model_prefix,
        filter_model_regexp=args.filter_model_regexp,
        no_base_model=args.no_base_model,
        filter_base_model=args.filter_base_model,
        filter_base_model_epoch=args.filter_base_model_epoch,
        trained=args.trained)

    if args.plot_runtime_vs_wer:
        plot_runtime_vs_wer_to_file(
            models_by_time=stats.models_by_time,
            filename=args.plot_runtime_vs_we,
            opts=eval("dict(%s)" % args.plot_runtime_vs_wer_opts))
        sys.exit(0)

    if args.best_model_setups:
        stats.print_best_model_times(number=args.best_model_setups)
        sys.exit(0)

    if args.best_models_and_train_scores:
        stats.print_best_models_and_train_scores()
        sys.exit(0)

    stats.print_gpu_time_factors()
    stats.print_best_model_times()
    stats.print_best_model_total_times()
    stats.print_best_models_by_time()


if __name__ == "__main__":
    main()
