#!/bin/bash
# Usage:         ./start-recog.sh $model $epoch
# Motivation:    1) It makes sure that
#                    - /data-train/model/ exists
#                    - there is a valid checkpoint in /data-train/experiment/net-model
#                    - and recog_dir/model/ is created.
#                2) Calls qint.py recog.q.sh -g 3 (-g 3 guarded with 3 tries) from recog_dir/model/
set -e

model=$1
epoch=$2
test -n "$model" && test -n "$epoch" || {
        echo "usage: $0 <model> <epoch>"
        exit 1
}

model_train_config=config-train/$model.config
test -e "$model_train_config" || {
        echo "$model_train_config does not exist"
        exit 1
}

model_train_dir=$(./tools-multisetup/_get_train_setup_dir.py $model_train_config || {
	echo "error calling ./tools-multisetup/_get_train_setup_dir.py $model_train_config" >/dev/stderr; exit 1; })
test -e "$model_train_dir" || {
        echo "$model_train_dir does not exist"
        exit 1
}

model_file_regexp="\(.*/network\.\(pretrain\.\)*0*$epoch\)\(\.meta\)*"
model_file="$(find $model_train_dir/net-model/ -regex "$model_file_regexp")"
test -n "$model_file" || {
	echo "$model_file_regexp does not exist"
	exit 1
}
model_file=$(echo "$model_file" | sed "s|$model_file_regexp|\1|g")
echo "model_file: $model_file"

recog_dir=data-recog/$model.${model_file##*.}
echo "recog_dir: $recog_dir"
test -e "$recog_dir" || {
	echo "$recog_dir does not exist"
	exit 1
}

cd $recog_dir
qint.py recog.q.sh -g 3
