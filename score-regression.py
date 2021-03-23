#!/usr/bin/env python3

"""
Reads in scores, tries to fit them to some simple model(Input: step, Output: score).
Usage:  python tools-multisetup/score-regression.py --score_file  scores/original.train.info.txt --score_key train_score
"""


import sys
import argparse
import re
import time
from typing import Dict
import numpy
import better_exchook


def parse_score_file(filename, score_key):
    """
    Parse score file with content like:
    ```
    epoch  19 dev_score_output/output_prob: 5.222746177016228
    epoch  20 dev_score_output/output_prob: 5.11850030781807
    epoch  21 dev_score_output/output_prob: 5.292179959449397
    epoch  22 dev_score_output/output_prob: 2.8004906318515395
    epoch  23 dev_score_output/output_prob: 1.89099183115423
    epoch  24 dev_score_output/output_prob: 1.4130756707448662
    epoch  25 dev_score_output/output_prob: 1.3210649910228496
    epoch  26 dev_score_output/output_prob: 1.3716357417908183
    ```

    Or RETURNN log file. Parse lines like:
    ```
    pretrain epoch 1, step 0, cost:ctc 89.24007017923032, cost:output/output_prob 9.21311959919558,
        error:ctc 6.523809549631551, error:decision 0.0, error:output/output_prob 1.000000003958121,
        loss 26877.72, max_size:classes 14, max_size:data 490, mem_usage:GPU:0 781.3MB, num_seqs 25,
        3.933 sec/step, elapsed 0:00:08, exp. remaining 0:00:09, complete 46.31%
    ```
    or
    ```
    pretrain epoch 1 'dev' eval, step 0, ...
    ```

    Or RETURNN eval score file in "py" format:
    ```
    {
    'train-clean-100-19-198-0000': {'seq_len': 6, 'score': 0.8934112, 'error': 0.33333334},
    ...
    }
    ```

    :param str filename:
    :param str|None score_key:
    :return: epoch|step -> score
    :rtype: dict[int,float]
    """
    scores_by_key = {}  # type: Dict[str,Dict[int,float]]  # key -> epoch|step -> score
    file_content = open(filename).read()
    if filename.endswith(".py") and file_content.startswith("{"):
        content = eval(file_content)
        assert isinstance(content, dict)
        seq_idx = 0
        for tag, values in sorted(content.items()):
            assert isinstance(values, dict)
            for key, value in values.items():
                scores_by_key.setdefault(key, {})[seq_idx] = value
            seq_idx += 1
    else:
        file_lines = file_content.splitlines()
        if any([s.startswith("RETURNN starting up") for s in file_lines[:20]]):
            assert score_key
            total_step = {}  # data_set -> int
            for line in file_lines:
                match = re.match(r"^(pretrain )?epoch ([0-9]+)( '([a-z]+)' eval)?, step ([0-9]+), (.*)$", line)
                if match:
                    _, epoch, _, data_set, step, other = match.groups()
                    epoch, step = int(epoch), int(step)
                    if not data_set:
                        data_set = "train"
                    other_parts = [item.split(" ", 1) for item in other.split(", ")]
                    scores = {
                        "%s_%s" % (data_set, key.replace("cost:", "score_").replace("error:", "error_")): float(value)
                        for (key, value) in other_parts
                        if key.startswith("cost:") or key.startswith("error:")}
                    total_step.setdefault(data_set, 0)
                    for key, value in scores.items():
                        scores_by_key.setdefault(key, {})[total_step[data_set]] = value
                    total_step[data_set] += 1
        else:
            if not score_key:
                score_key = ""  # e.g. WER
            for line in file_lines:
                match = re.match(r"^epoch\s+([0-9]+) ?(.*): ([0-9.]+)$", line)
                if not match:
                    continue
                epoch, key, score = match.groups()
                epoch, key, score = int(epoch), key, float(score)
                scores_by_key.setdefault(key, {})[epoch] = score
    assert scores_by_key
    if score_key in scores_by_key:
        return scores_by_key[score_key]
    matching_keys = [key for key in scores_by_key if key.startswith(score_key)]
    assert len(matching_keys) == 1, "score keys %r, no unique match to %r, got %r" % (
        list(scores_by_key.keys()), score_key, matching_keys)
    return scores_by_key[matching_keys[0]]


class LivePlot:
    def __init__(self, scores, ref_points_format="b-"):
        """
        :param dict[int,float] scores:
        :param str ref_points_format:
        """
        from matplotlib import pyplot as plt
        x_min, x_max = min(scores), max(scores)
        x_min -= (x_max - x_min) * 0.1
        x_max += (x_max - x_min) * 0.5
        self.x_min_max = (x_min, x_max)
        y_min, y_max = min(scores.values()), max(scores.values())
        if y_min > 0:
            y_min = 0
        y_max += (y_max - y_min) * 0.1
        self.y_min_ax = (y_min, y_max)
        fig, ax = plt.subplots(1, 1)
        self._fig = fig
        self._ax = ax
        # ax.set_aspect('equal')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        plt.show(block=False)
        plt.draw()


        # cache the background
        # self._background = fig.canvas.copy_from_bbox(ax.bbox)
        fig.canvas.draw()

        x_np = sorted(scores.keys())
        y_np = [scores[x_] for x_ in x_np]
        x_np, y_np = numpy.array(x_np).astype("float32"), numpy.array(y_np).astype("float32")

        self._ref_points = ax.plot(x_np, y_np, ref_points_format)[0]
        self._points = ax.plot([], [], "r-")[0]

    def draw(self, x=None, y=None):
        """
        :param numpy.ndarray|None x:
        :param numpy.ndarray|None y:
        """
        from matplotlib import pyplot as plt
        # update the xy data
        if x is not None and y is not None:
            self._points.set_data(x, y)

        # restore background
        # self._fig.canvas.restore_region(self._background)

        # redraw just the points
        self._ax.draw_artist(self._ref_points)
        self._ax.draw_artist(self._points)

        # fill in the axes rectangle
        self._fig.canvas.blit(self._ax.bbox)

        plt.pause(0.0001)




def model(x, bias_init=0.0):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    def _positive(x):
        """
        :param tf.Tensor x:
        :rtype: tf.Tensor
        """
        return tf.abs(x)


    def exp_model(x):
        """
        :param tf.Tensor x:
        :rtype: tf.Tensor
        """
        with tf.variable_scope("exp_model"):
            a = tf.get_variable(name="a", shape=(), initializer=tf.constant_initializer(1.0))
            scale = tf.get_variable(name="scale", shape=(), initializer=tf.constant_initializer(1.0))
            return tf.exp(-x * a) * _positive(scale)


    def exp_exp_model(x):
        """
        :param tf.Tensor x:
        :rtype: tf.Tensor
        """
        with tf.variable_scope("exp_exp_model"):
            a = tf.get_variable(name="a", shape=(), initializer=tf.constant_initializer(1.0))
            b = tf.get_variable(name="b", shape=(), initializer=tf.constant_initializer(1.0))
            scale = tf.get_variable(name="scale", shape=(), initializer=tf.constant_initializer(1.0))
            return tf.exp(tf.exp(-x * a) * b) * _positive(scale)


    def sigmoid_model(x):
        """
        :param tf.Tensor x:
        :rtype: tf.Tensor
        """
        with tf.variable_scope("sigmoid_model"):
            a = tf.get_variable(name="a", shape=(), initializer=tf.constant_initializer(1.0))
            b = tf.get_variable(name="b", shape=(), initializer=tf.constant_initializer(0.0))
            c = tf.get_variable(name="c", shape=(), initializer=tf.constant_initializer(1.0))
            d = tf.get_variable(name="d", shape=(), initializer=tf.constant_initializer(1.0))
            scale = tf.get_variable(name="scale", shape=(), initializer=tf.constant_initializer(1.0))
            return tf.sigmoid(a - b * x - tf.sqrt(_positive(x * c + d))) * _positive(scale)


    def log_model(x):
        """
        :param tf.Tensor x:
        :rtype: tf.Tensor
        """
        with tf.variable_scope("log_model"):
            a = tf.get_variable(name="a", shape=(), initializer=tf.constant_initializer(1.0))
            b = tf.get_variable(name="b", shape=(), initializer=tf.constant_initializer(0.0))
            scale = tf.get_variable(name="scale", shape=(), initializer=tf.constant_initializer(1.0))
            return -tf.log(_positive(x * a + b + 1.0)) * _positive(scale)


    def log_log_model(x):
        """
        :param tf.Tensor x:
        :rtype: tf.Tensor
        """
        with tf.variable_scope("log_log_model"):
            a = tf.get_variable(name="a", shape=(), initializer=tf.constant_initializer(1.0))
            b = tf.get_variable(name="b", shape=(), initializer=tf.constant_initializer(0.0))
            c = tf.get_variable(name="c", shape=(), initializer=tf.constant_initializer(1.0))
            d = tf.get_variable(name="d", shape=(), initializer=tf.constant_initializer(0.0))
            scale = tf.get_variable(name="scale", shape=(), initializer=tf.constant_initializer(1.0))
            return -tf.log(_positive(-tf.log(_positive(x * a + b) + 1.0) * c + d)) * _positive(scale)


    def neural_model(x):
        """
        :param tf.Tensor x:
        :rtype: tf.Tensor
        """
        with tf.variable_scope("neural_model"):
            y = tf.expand_dims(x, axis=1)  # (batch,dim)
            y = tf.keras.layers.Dense(name="l1", activation=tf.nn.relu, units=3)(y)
            y = tf.keras.layers.Dense(name="l2", activation=tf.nn.relu, units=3)(y)
            y = tf.keras.layers.Dense(name="l3", activation=tf.nn.relu, units=3)(y)
            w = tf.get_variable(name="w", shape=(y.shape[-1].value,), initializer=tf.glorot_uniform_initializer())
            return tf.reduce_sum(w * y, axis=1)


    """
    :param tf.Tensor x: (batch,)
    :param float bias_init:
    :return: y, (batch,)
    :rtype: tf.Tensor
    """
    bias = tf.get_variable(name="bias", shape=(), initializer=tf.constant_initializer(bias_init))
    y = bias
    # y = y + exp_model(x)
    y = y + neural_model(x)
    # y = y + exp_exp_model(x)
    # y = y + sigmoid_model(x)
    # y = y + log_model(x)
    # y = y + log_log_model(x)
    return y


def fit(scores, max_num_steps=float("inf"), verbose=True, plot=None):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    """
    :param dict[int,float] scores:
    :param int|float max_num_steps:
    :param bool verbose:
    :param LivePlot|None plot:
    :return: final loss, final params
    :rtype: (float, dict[str,float])
    """
    print("Got %i scores." % len(scores))
    x_np = sorted(scores.keys())
    # x_np = x_np[:len(x_np) // 2]
    y_np = [scores[x_] for x_ in x_np]
    x_np, y_np = numpy.array(x_np).astype("float32"), numpy.array(y_np).astype("float32")
    x_norm = x_np[-1] - x_np[0]
    x_np /= x_norm
    tf.set_random_seed(42)
    x = tf.placeholder(tf.float32, name="x", shape=(None,))
    y = tf.placeholder(tf.float32, name="y", shape=(None,))
    y_ = model(x, bias_init=min(y_np))
    diff = tf.squared_difference(y, y_)
    # diff = tf.exp((y - y_) ** 2)
    loss = tf.reduce_mean(diff)
    params = {v.name[:-2]: v for v in tf.trainable_variables()}
    params_l2 = tf.reduce_sum([tf.nn.l2_loss(v) for v in params.values()])
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        x_range = tf.range(-0.5, 1.5, 2. / len(x_np))
        y_range = model(x_range)
        # We want that the model is monotonically decreasing.
        dx, = tf.gradients(y_range, x_range)
        # monotonic_loss = tf.maximum(y_range[1:] - y_range[:-1], 0.0) ** 2.0
        monotonic_loss = tf.maximum(dx, 0.0) ** 2.0
    loss_with_reg = loss + params_l2 * 0.001 + monotonic_loss * 10.0
    # opt = tf.train.RMSPropOptimizer(learning_rate=0.01)
    opt = tf.train.AdamOptimizer(learning_rate=0.01)
    update = opt.minimize(loss_with_reg)
    print("num params:", numpy.sum([numpy.prod([v.get_shape().as_list() or [1]]) for v in params.values()]))
    step = 0
    next_print_step = 10
    next_plot_step = 10
    model_x_np = numpy.arange(-0.1, 1.5, step=0.001) * x_norm + x_np[0]
    with tf.Session() as session:
        session.run(tf.variables_initializer(tf.global_variables()))
        while step < max_num_steps:
            loss_np, _, params_np = session.run((loss, update, params), feed_dict={x: x_np, y: y_np})
            if verbose and step % (next_print_step // 10) == 0:
                print("step:", step, "loss:", loss_np, "params:", params_np)
                if step >= next_print_step:
                    next_print_step *= 10
            if plot and step % (next_plot_step // 10) == 0:
                model_y_np = session.run(y_, feed_dict={x: model_x_np / x_norm})
                plot.draw(model_x_np, model_y_np)
                if step >= next_plot_step and next_plot_step < 1000:
                    next_plot_step *= 10
            step += 1
        return loss_np, params_np


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--score_dir")
    arg_parser.add_argument("--score_file")
    arg_parser.add_argument("--score_key")
    arg_parser.add_argument("--sort_scores", action="store_true")
    arg_parser.add_argument("--no_fit", action="store_true")
    args = arg_parser.parse_args()
    if bool(args.score_dir) == bool(args.score_file):
        print("Provide either score_dir or score_file.")
        arg_parser.print_usage()
        sys.exit(1)
    if args.score_file:
        # 1.Parse
        scores = parse_score_file(filename=args.score_file, score_key=args.score_key)
        if args.sort_scores:
            scores_values = numpy.array(list(scores.values()))
            assert scores_values.shape == (len(scores),)
            # 2.Sort
            scores_values.sort()
            scores = {i: x for (i, x) in enumerate(scores_values)}
        # 3.Plot
        plot = LivePlot(scores=scores)
        try:
            if args.no_fit:
                while True:
                    plot.draw()
            else:
                try:
                    # 4.Fit
                    fit(scores, plot=plot)
                except KeyboardInterrupt:
                    print("Got KeyboardInterrupt. Press again to quit.")
                    while True:
                        plot.draw()
        except KeyboardInterrupt:
            print("Got KeyboardInterrupt, quit.")
            sys.exit(1)
        sys.exit()
    if args.score_dir:
        raise NotImplementedError


if __name__ == "__main__":
    better_exchook.install()
    main()
