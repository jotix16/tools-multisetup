Tools for multi setups, where there are multiple train configs and we can run multiple trains and recogs in parallel.

This tools intend to automize runnning different task in RETURNN such as:
- training
- recognition
- cleaning
- ..
# Documentation of the scripts

## tools-multisetup
- [_get_train_setup_dir.py](_get_train_setup_dir.py) -- Prints corresponding path of working dir `data-train/$experiment/` for the given experiment config path. Used in shell scripts.
- [cleanup-all-except.py](cleanup-all-except.py) -- Cleans all setups except the ones given as arguments. For cleaning it calls `reset-train.py` and than deletes the config under config-train/*.config.
- [cleanup-old-models.py](cleanup-old-models.py) -- If use `--doit` it cleans otherwise prints old models in `data-train/model/net-model` that can be cleaned.
- [ ] [collect-lur-stats.py](collect-lur-stats.py)
- [ ] [auto-recog.py](auto-recog.py)
- [create-recog.sh](create-recog.sh) \$**experiment** \$**epoch** -- Creates a strucuture under data-recog with folders for each epoch the recognition is performed.
- [qstat-recog.sh](qstat-recog.sh) --- !!! WEIRD command pqstat !!!
- [start-recog.sh](start-recog.sh) -- Makes sure the files in data-train and data-recog exist and Call `qint.py recog.q.sh -g 3` from `recog_dir/model/`
- [create-train.py](create-train.py) \$**experiment** -- Creates the structure of data-train where the data for different experiments are saved. (net-mode, log, qlog)
- [create-train.sh](create-train.sh) \$**experiment** -- Calls [create-train.py](create-train.py)
- [start-train.sh](start-train.sh) \$**experiment** -- Calls `qint.py train.q.sh -g 3` from the `data-train/experiment`
- [qstat-train.sh](qstat-train.sh) -- !!! WEIRD command pqstat !!!
- [reset-train.py](reset-train.py) -- Stops training for $experiment, deletes `data-train/$experiment` and removes the corresponding files in `scores/*` and `logs-archive/*`.
- [stop-train.py](stop-train.py) \$**experiment** -- Calls qdel after finding the jobid for the \$**experiment**.
- [get-best-train-info-scores.py](get-best-train-info-scores.py) -- It extracts score information from score files, e.g. epoch with max score. If no file given it checks all files for the specified --key score_type. If file specified it prints information about all score_types for that file. With score type we mean the the name of the dataset the score is calculated on, e.g. devtrain_score, train_score etc.
- [ ] [extract-scores.py](extract-scores.py)
- [mark-finished-train.py](mark-finished-train.py) -- Motivation: Used to mark a setup config as finished with a comment # multistep: $reason
- [rm-old-train-models.py](rm-old-train-models.py) -- Filter out N best setups in `net-model/` - don't delete anything from them. From the remaining, delete all but the best epoch.
- [get-status.py](get-status.py) -- Prints informaion about finished & incomplete running/not-running training and running jobs.
- [score-regression.py](score-regression.py) -- Reads in scores, tries to fit them to some simple model
- [show-log.py](show-log.py) -- !! do a link to tools-multistep rather than copy.!! Simplifies navigating through the qlogs in `qdir`. It calls `$less $fn`for either recog(in `data-recog/`) or train(in `data-train/`) qlog files.
- [time-stats.py](time-stats.py) -- Allows you to collect time related information about trained models and maybe plot runtime vs wer.
- [ ] [tools.py](tools.py)


## Folders
- [ ] [crnn@](crnn) [->]() ../returnn
- [ ] [libs@](libs) -- tools to work with shell
- [ ] [i6lib@](i6lib) [->]() libs/i6lib
- [ ] [lib@](lib) [->]() libs/lib

