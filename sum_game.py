import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.serialization import save
from torch.utils.data import DataLoader, Dataset

import egg.core as core
from egg.core import Callback, Interaction, PrintValidationEvents


# the following section specifies parameters that are specific to the sum games: this also inherits the
# standard EGG parameters from https://github.com/facebookresearch/EGG/blob/main/egg/core/util.py
def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data", type=str, default=None, help="Path to the train data"
    )
    parser.add_argument(
        "--validation_data", type=str, default=None, help="Path to the validation data"
    )
    parser.add_argument(
        "--n_terms",
        type=int,
        default=None,
        help="Number of input integers to be summed",
    )
    parser.add_argument(
        "--n_values",
        type=int,
        default=None,
        help="Number of integers in range",
    )
    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=0,
        help="Batch size when processing validation data, whereas training data batch_size is controlled by batch_size (default: same as training data batch size)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="rf",
        choices=["rf", "gs"],
        help="Selects whether Reinforce or Gumbel-Softmax relaxation is used for training {rf, gs} (default: rf)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="GS temperature for the sender, only relevant in Gumbel-Softmax (gs) mode (default: 1.0)",
    )
    parser.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=1e-1,
        help="Reinforce entropy regularization coefficient for Sender, only relevant in Reinforce (rf) mode (default: 1e-1)",
    )
    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Receiver (default: 10)",
    )
    # arguments controlling the script output
    parser.add_argument(
        "--train_results_path",
        type=str,
        help="Path JSON file to store results from training set",
    )
    parser.add_argument(
        "--val_results_path",
        type=str,
        help="Path JSON file to store results from validation set",
    )
    args = core.init(parser, params)
    return args

class SaveResults(Callback):
    def __init__(self, train_results, val_results, n_epochs, n_values):
        super().__init__()
        self.train_results = train_results
        self.val_results = val_results
        self.n_epochs = n_epochs
        self.n_values = n_values

    @staticmethod
    def save_events(logs, save_path, n_values):
        with open(save_path, 'w') as f:
            json.dump({
                "inputs": [
                    [x - i * n_values for i, x in enumerate((m > 0).nonzero(as_tuple=True)[0].tolist())] 
                    for m in logs.sender_input
                    ],
                "labels": [m.tolist() for m in logs.labels],
                "messages": [m.tolist() for m in logs.message],
                "outputs": [m.argmax().tolist() for m in logs.receiver_output]
            }, f)

    # here is where we make sure we are printing the validation set (on_validation_end, not on_epoch_end)
    def on_validation_end(self, _loss, logs, epoch: int):
        # here is where we check that we are at the last epoch
        if epoch == self.n_epochs:
            self.save_events(logs, self.val_results, self.n_values)
    
    def on_epoch_end(self, _loss, logs, epoch: int):
        # here is where we check that we are at the last epoch
        if epoch == self.n_epochs:
            self.save_events(logs, self.train_results, self.n_values)

class SumCategoricalDataset(Dataset):
    def __init__(self, path, n_terms, n_values):
        self.frame = []

        with open(path, 'r') as f:
            for row in f:
                row = row.split()
                config = list(map(int, row))

                # store each number as a one-hot vector, where the xth element is 1 for input x
                z = torch.zeros((n_terms, n_values))
                for i in range(n_terms):
                    z[i, config[i]] = 1
                label = torch.tensor(sum(config))

                # Flatten stack of input elements into input vector
                self.frame.append((z.view(-1), label))

    def get_n_features(self):
        return self.frame[0][0].size(0)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]

class SumCategoricalReceiver(nn.Module):
    def __init__(self, n_hidden, n_outputs):
        super(SumCategoricalReceiver, self).__init__()

        # 2 hidden layer MLP receiver, acting on embeddings of dimension n_hidden
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_outputs)

    def forward(self, x, _input=None, _aux_input=None):
        return self.fc2(
                torch.tanh(self.fc1(x))
                )

class SumCategoricalSender(nn.Module):
    def __init__(self, n_hidden, n_features, n_vocab):
        super(SumCategoricalSender, self).__init__()

        # 2 hidden layer MLP sender acting on input features
        self.fc1 = nn.Linear(n_features, n_hidden)
        # self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_vocab)

    def forward(self, x, _aux_input=None):
        return F.log_softmax(
            self.fc3(
                # torch.tanh(self.fc2(
                        torch.tanh(self.fc1(x))
                    # ))
            ), dim=1)

def main(params):
    opts = get_params(params)
    if opts.validation_batch_size == 0:
        opts.validation_batch_size = opts.batch_size
    print(opts, flush=True)

    def loss(
        sender_input, _message, _receiver_input, receiver_output, labels, _aux_input
    ):
        n_terms = opts.n_terms
        n_values = opts.n_values
        batch_size = sender_input.size(0)

        # The sum of n_terms numbers, each of them between 0 and n_values - 1
        # is between 0 and n_terms * (n_values - 1)
        receiver_output = receiver_output.view(batch_size, n_terms * (n_values - 1) + 1)
        receiver_guesses = receiver_output.argmax(dim=1)
        correct_samples = ((receiver_guesses == labels).detach())
        acc = correct_samples.float()
        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        return loss, {"acc": acc}

    train_loader = DataLoader(
        SumCategoricalDataset(
            path=opts.train_data,
            n_terms=opts.n_terms,
            n_values=opts.n_values,
        ),
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=1,
    )
    test_loader = DataLoader(
        SumCategoricalDataset(
            path=opts.validation_data,
            n_terms=opts.n_terms,
            n_values=opts.n_values,
        ),
        batch_size=opts.validation_batch_size,
        shuffle=False,
        num_workers=1,
    )

    receiver = SumCategoricalReceiver(
        n_hidden=opts.receiver_hidden, 
        n_outputs=opts.n_terms * (opts.n_values - 1) + 1
    )

    # the number of input features for the the Sender is given by n_terms * n_values because
    # they are fed 1-hot representations of the input numbers
    n_features = opts.n_terms * opts.n_values

    sender = SumCategoricalSender(n_hidden=opts.sender_hidden, n_features=n_features, n_vocab=opts.vocab_size) 

    # We wrap the sender and receiver networks in the respective wrappers based on the training method
    if opts.mode.lower() == "gs":
        sender = core.GumbelSoftmaxWrapper(sender)
        receiver = core.SymbolReceiverWrapper(receiver, opts.vocab_size, opts.receiver_hidden)
        game = core.SymbolGameGS(sender, receiver, loss)
        callbacks = [core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1)]
    else: # defaults to REINFORCE
        sender = core.ReinforceWrapper(sender)

        # First wrap the agent in a receiver wrapper (which embeds the message) and then in the
        # ReinforceDeterministicWrapper for computing the output during REINFORCE training
        receiver = core.ReinforceDeterministicWrapper(
            core.SymbolReceiverWrapper(receiver, opts.vocab_size, opts.receiver_hidden)
            )
        game = core.SymbolGameReinforce(
            sender,
            receiver,
            loss,
            sender_entropy_coeff=opts.sender_entropy_coeff,
            receiver_entropy_coeff=0,
        )
        callbacks = []

    optimizer = core.build_optimizer(game.parameters())
    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=test_loader,
        callbacks=callbacks
        + [
            core.ConsoleLogger(print_train_loss=True, as_json=True),
            SaveResults(opts.train_results_path, opts.val_results_path, opts.n_epochs, opts.n_values)
        ],
    )

    trainer.train(n_epochs=opts.n_epochs)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
