from __future__ import print_function
import pickle as pkl
import numpy as np
import logging


# Seed a random number generator
seed = 10102016
rng = np.random.RandomState(seed)
batch_size = 100
# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# The model set up code below is provided as a starting point.
# You will probably want to add further code cells for the
# different experiments you run.
from mlp.helper import train_model_and_plot_stats
from mlp.data_providers import EMNISTDataProvider
from mlp.layers import *
from mlp.errors import CrossEntropyLogSoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule, AdamLearningRule, RMSPropLearningRule
from mlp.penalty import *

# Create data provider objects for the MNIST data set
train_data = EMNISTDataProvider('train', batch_size=batch_size, rng=rng)
valid_data = EMNISTDataProvider('valid', batch_size=batch_size, rng=rng)
test_data = EMNISTDataProvider('test', batch_size=batch_size, rng=rng)

# setup hyperparameters
learning_rate = 0.1
num_epochs = 150
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 100

trial=1
train_data.reset()
valid_data.reset()
test_data.reset()

weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

model = MultipleLayerModel([
    DropoutLayer(rng=rng, incl_prob=.8, share_across_batch=True),
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
        kernels_penalty=L2Penalty(1e-6)),
    MaxPoolingLayer(
        num_input_channels=5, input_dim_1=24, input_dim_2=24, extent=2),
    ReluLayer(),
    ConvolutionalLayer(
        num_input_channels=5,
        num_output_channels=10,
        input_dim_1=12,
        input_dim_2=12,
        kernel_dim_1=5,
        kernel_dim_2=5,
        padding=0,
        stride=1,
        kernels_penalty=L2Penalty(1e-6)),
    MaxPoolingLayer(
        num_input_channels=10, input_dim_1=8, input_dim_2=8, extent=2),
    ReluLayer(),
    ReshapeLayer(output_shape=(4 * 4 * 10,)), # TWO HIDDEN LAYER
    DropoutLayer(rng=rng, incl_prob=.8, share_across_batch=True),
    AffineLayerWithoutBias(4*4*10, 400, weights_init, weights_penalty=L2Penalty(1e-6)),
    BatchNormalizationLayer(input_dim=(400), rng=rng),
    ReluLayer(),
    DropoutLayer(rng=rng, incl_prob=.8, share_across_batch=True),
    AffineLayerWithoutBias(400, 400, weights_init, weights_penalty=L2Penalty(1e-6)),
    BatchNormalizationLayer(input_dim=(400), rng=rng),
    ReluLayer(),
    AffineLayer(400, output_dim, weights_init, biases_init,weights_penalty=L2Penalty(1e-6))
])

print(model)
error = CrossEntropyLogSoftmaxError()
learning_rule = AdamLearningRule(learning_rate=1e-3) # Increase because of BN

# Remember to use notebook=False when you write a script to be run in a terminal
output = train_model_and_plot_stats(model,
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

pkl.dump(output, open('2conv_2DNN_BN_DROPOUT_REG_ADAM_1e-3.pkl', 'wb'), protocol=-1)
