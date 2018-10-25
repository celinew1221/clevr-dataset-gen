#!/usr/bin/env bash

truncate -s 0 $4

blender --background --python render_images.py -- --use_gpu $3 --start_idx $1 --num_images $2 --log_file $4 @args

start=$(tail -n 1 $4)
rest=$(( $2 - $start + $1 ))

while [[ $rest > 0 ]]
do
    blender --background --python render_images.py -- --use_gpu $3 --start_idx $start --num_images $rest --log_file $4 @args
    start=$(tail -n 1 $4)
    rest=$(( $2 - $start + $1 ))
done
