from __future__ import print_function

from mlp.helper import train_model_and_plot_stats
# The below code will set up the data providers, random number
# generator and logger objects needed for training runs. As
# loading the data from file take a little while you generally
# will probably not want to reload the data providers on
# every training run. If you wish to reset their state you
# should instead use the .reset() method of the data providers.
import numpy as np
import logging
from mlp.data_providers import MNISTDataProvider, EMNISTDataProvider

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

# The model set up code below is provided as a starting point.
# You will probably want to add further code cells for the
# different experiments you run.

from mlp.layers import *
from mlp.errors import CrossEntropyLogSoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule, AdamLearningRule, RMSPropLearningRule
from mlp.optimisers import Optimiser, EarlyStoppingOptimiser
from mlp.penalty import *

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
    DropoutLayer(rng=rng, incl_prob=0.8, share_across_batch=True),
    ReshapeLayer(output_shape=(1, 28, 28)),
    ConvolutionalLayer(
        num_input_channels=1,
        num_output_channels=5,
        input_dim_1=28,
        input_dim_2=28,
        kernel_dim_1=5,
        kernel_dim_2=5,
        padding=0,
        stride=1,
        kernels_penalty=L2Penalty(1e-5)),
    MaxPoolingLayer(
        num_input_channels=5, input_dim_1=24, input_dim_2=24, extent=2),
    ReshapeLayer(),
    AffineLayer(12*12*5, 400,weights_init, biases_init, L2Penalty(1e-5)),
    DropoutLayer(rng=rng, incl_prob=.8, share_across_batch=True),
    ReluLayer(),
    AffineLayer(400, output_dim, weights_init, biases_init, L2Penalty(1e-5)),
])

error = CrossEntropyLogSoftmaxError()
learning_rule = GradientDescentLearningRule(learning_rate=.1)

# Remember to use notebook=False when you write a script to be run in a terminal
trial1 = train_model_and_plot_stats(model,
                                    error,
                                    learning_rule,
                                    train_data,
                                    valid_data,
                                    test_data,
                                    num_epochs,
                                    stats_interval,
                                    notebook=False,
                                    displayGraphs=False,
                                    earlyStop=True,
                                    steps=3,
                                    patience=5)

import pickle as pkl

pkl.dump(trial1, open('convNet_1layer_wDropout2.pkl', 'wb'), protocol=-1)
