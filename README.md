# Sum communication game

## Prerequisites
Create a virtual environment/`conda` environment and install the [EGG](https://github.com/facebookresearch/EGG]) library.

## Running
For a simple game where the sender has to communicate the sum of 2 numbers to another agent, we first generate the data and store it in text files, and then pass these to the training script with the appropriate hyperparameters.

To generate data for range $N$,
```
for i in `seq 0 <N>`; do for j in `seq 0 <N - 1>`; do echo $i $j; done; done | shuf > dataset.txt
```

To split into disjoint train and test sets,
```
head -n <N_train> dataset.txt > train.txt
tail -n <N_test> dataset.txt > test.txt
```
To ensure there is no train-test overlap, ensure that $N_{train} + N_{test} \le N_{dataset}$.

To train the models,
```
python sum_game.py \
    --mode gs \
    --train_data train.txt \
    --validation_data test.txt \
    --n_terms 2 --n_values N --vocab_size V \
    --n_epochs N_EPOCHS --batch_size BATCH_SIZE --lr LR \
    --validation_batch_size VALIDATION_BATCH_SIZE --validation_freq VALIDATION_FREQ \
    --sender_hidden 128 --receiver_hidden 128 \
    --random_seed 42  \
    --train_results_path train_results.json --val_results_path val_results.json
```
This will train the models and store the language analysis of training samples in `train_results.json` and that of test samples in `val_results.json`.

Hyperparameters used for the experiments in the report are shown in [`run.sh`](./run.sh). Splits used for Commutativity, Commutativity_0, Identity, and Identity_0 settings are available in [`/data`](./data)