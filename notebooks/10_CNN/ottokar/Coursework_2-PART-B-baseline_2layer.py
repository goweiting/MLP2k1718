from __future__ import print_function


def train_model_and_plot_stats(model,
                               error,
                               learning_rule,
                               train_data,
                               valid_data,
                               test_data,
                               num_epochs,
                               stats_interval,
                               notebook=False,
                               earlyStopping=True):
    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors = {'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    # Use the created objects to initialise a new Optimiser instance.
    if earlyStopping:
        optimiser = EarlyStoppingOptimiser(
            model,
            error,
            learning_rule,
            train_data,
            valid_data,
            test_data,
            data_monitors,
            notebook=notebook,
            steps=3,
            patience=3)
        stats, key, run_time, best_epoch = optimiser.train(
            max_num_epochs=num_epochs, stats_interval=stats_interval)
        return (optimiser.model, stats, keys, run_time,best_epoch)
    else:
        optimiser = Optimiser(
            model,
            error,
            learning_rule,
            train_data,
            valid_data,
            test_data,
            data_monitors,
            notebook=notebook)

        stats, keys, run_time = optimiser.train(
            num_epochs=num_epochs, stats_interval=stats_interval)
        return (optimiser.model, stats, keys, run_time)

import numpy as np
import logging
from mlp.data_providers import EMNISTDataProvider

# Seed a random number generator
seed = 10102016
rng = np.random.RandomState(seed)
batch_size = 100
# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = EMNISTDataProvider('train', batch_size=batch_size, rng=rng)
valid_data = EMNISTDataProvider('valid', batch_size=batch_size, rng=rng)
test_data = EMNISTDataProvider('test', batch_size=batch_size, rng=rng)

from mlp.layers import *
from mlp.errors import CrossEntropyLogSoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule, AdamLearningRule, RMSPropLearningRule, MomentumLearningRule
from mlp.optimisers import Optimiser, EarlyStoppingOptimiser

# setup hyperparameters
learning_rate = 0.1
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 100

train_data.reset()
valid_data.reset()
test_data.reset()

weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)
model = MultipleLayerModel([
    ReshapeLayer(output_shape=(1, 28, 28)),
    ConvolutionalLayer(
        num_input_channels=1,
        num_output_channels=5,
        input_dim_1=28,
        input_dim_2=28,
        kernel_dim_1=5,
        kernel_dim_2=5,
        padding=0,
        stride=1),
    MaxPoolingLayer(
        num_input_channels=5, input_dim_1=24, input_dim_2=24, extent=2),
    ReshapeLayer(output_shape=(12 * 12 * 5,)),
    ReluLayer(),
    ReshapeLayer(output_shape=(5,12,12)),
    ConvolutionalLayer(
        num_input_channels=5,
        num_output_channels=10,
        input_dim_1=12,
        input_dim_2=12,
        kernel_dim_1=5,
        kernel_dim_2=5,
        padding=0,
        stride=1),
    MaxPoolingLayer(
        num_input_channels=10, input_dim_1=8, input_dim_2=8, extent=2),
    ReshapeLayer(output_shape=(4 * 4 * 10,)),
    AffineLayer(4*4*10, output_dim, weights_init, biases_init),
    ReluLayer(),
    AffineLayer(output_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropyLogSoftmaxError()
learning_rule = MomentumLearningRule()

# Remember to use notebook=False when you write a script to be run in a terminal
trial1 = train_model_and_plot_stats(
    model,
    error,
    learning_rule,
    train_data,
    valid_data,
    test_data,
    num_epochs,
    stats_interval,
    notebook=False,
    earlyStopping=False)

import pickle as pkl

pkl.dump(trial1, open('baseline2_NOEARLYSTOP_MOMENTUM.pkl', 'wb'), protocol=-1)
