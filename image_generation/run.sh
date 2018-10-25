#!/usr/bin/env bash

truncate -s 0 log.log

GPU="0 1"
blender --background --python render_images.py -- --use_gpu $GPU --start_idx $1 --num_images $2 @args

start=`tail -n 1 log.log`
rest=$(( $2 - $start ))

while [[ $rest > 0 ]]
do
    blender --background --python render_images.py -- --use_gpu $GPU --start_idx $start --num_images $rest @args
    start=`tail -n 1 log.log`
    rest=$(( $2 - $start ))
done