#!/bin/bash

set -e

cd $(dirname $0)
mydir=$(realpath .)

model=$1
epoch=$2
test -n "$model" && test -n "$epoch" || {
	echo "usage: $0 <model> <epoch>"
	exit 1
}

model_train_config=config-train/$model.config
test -e "$model_train_config" || {
	echo "config $model_train_config does not exist"
	exit 1
}

model_train_dir=$(./tools-multisetup/_get_train_setup_dir.py $model_train_config)
test -e "$model_train_dir" || {
	echo "model train dir $model_train_dir does not exist"
	exit 1
}

model_file_regexp="\(.*/network\.\(pretrain\.\)*0*$epoch\)\(\.meta\)*"
model_file="$(find $model_train_dir/net-model/ -regex "$model_file_regexp")"
test -n "$model_file" || {
	echo "model file $model_file_regexp does not exist"
	exit 1
}
model_file=$(echo "$model_file" | sed "s|$model_file_regexp|\1|g")
echo "model_file: $model_file"

recog_dir=data-recog/$model.${model_file##*.}

echo "creating $recog_dir"
mkdir $recog_dir
mkdir $recog_dir/qdir
mkdir $recog_dir/log
mkdir $recog_dir/log.opt
mkdir $recog_dir/data-recog
mkdir $recog_dir/ctms
mkdir $recog_dir/lattices
mkdir $recog_dir/scoring
cp recog.q.sh $recog_dir/
ln -snf $mydir $recog_dir/base
ln -snf $mydir $recog_dir/setup-basedir
ln -s setup-basedir/recog.settings.sh $recog_dir/
test -e $mydir/theano-cpu-activate.sh &&  ln -s setup-basedir/theano-cpu-activate.sh $recog_dir/
ln -s setup-basedir/data-train $recog_dir/
ln -s setup-basedir/tools  $recog_dir/
ln -s setup-basedir/tools-recog $recog_dir/
ln -s setup-basedir/flow $recog_dir/
ln -s setup-basedir/flow-recog $recog_dir/
ln -s setup-basedir/features $recog_dir/
ln -s setup-basedir/features.warped $recog_dir/
ln -s setup-basedir/features.recog $recog_dir/
ln -s setup-basedir/features.recog.warped $recog_dir/
ln -s setup-basedir/config $recog_dir/
ln -s setup-basedir/config-train $recog_dir/
ln -s setup-basedir/crnn $recog_dir/
ln -s setup-basedir/dependencies $recog_dir/
ln -s setup-basedir/sprint-executables $recog_dir/
ln -snf setup-basedir/$model_train_config $recog_dir/train.crnn.config
ln -s setup-basedir/$model_train_dir/net-model $recog_dir/

echo "model=$model" > $recog_dir/settings.sh
echo "model_file=$model_file" >> $recog_dir/settings.sh
echo "epoch=$epoch" >> $recog_dir/settings.sh
echo "extra_recog_options=\"$(./tools-multisetup/_get_extra_recog_options.py $model_train_config $epoch)\"" >> $recog_dir/settings.sh
