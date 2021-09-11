#!/bin/bash

VALUES=$1
VOCAB=$2
SETTING=$3

if [[ $VALUES -eq 10 ]]
then
    LEARNING_RATE=0.005
    BATCH_SIZE=8
    SENDER_HIDDEN=64
    RECEIVER_HIDDEN=64
    if [[ $SETTING = "comm_0" ]]
    then
        N_EPOCHS=1500
    else
        N_EPOCHS=1000
    fi
fi

if [[ $VALUES -eq 15 ]]
then
    LEARNING_RATE=0.005
    BATCH_SIZE=10
    N_EPOCHS=2500
    SENDER_HIDDEN=64
    RECEIVER_HIDDEN=64
fi

if [[ $VALUES -eq 25 ]]
then
    LEARNING_RATE=0.005
    BATCH_SIZE=50
    N_EPOCHS=2500
    SENDER_HIDDEN=128
    RECEIVER_HIDDEN=128
fi

python sum_game.py \
    --mode rf \
    --train_data data/sum_${VALUES}_train_${SETTING}.txt \
    --validation_data data/sum_${VALUES}_test_${SETTING}.txt \
    --n_terms 2 --n_values ${VALUES} --vocab_size ${VOCAB} \
    --n_epochs $BATCH_SIZE --batch_size $BATCH_SIZE --lr $LEARNING_RATE \
    --validation_batch_size 50 --validation_freq 25 \
    --sender_hidden $SENDER_HIDDEN --receiver_hidden $RECEIVER_HIDDEN \
    --random_seed 42  \
    --train_results_path results/sum_values=${VALUES}_vocab=${VOCAB}_setting=${SETTING}_train.json \
    --val_results_path results/sum_values=${VALUES}_vocab=${VOCAB}_setting=${SETTING}_val.json