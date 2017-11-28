from __future__ import print_function
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pickle as pkl

def train_model_and_plot_stats(model,
                               error,
                               learning_rule,
                               train_data,
                               valid_data,
                               test_data,
                               num_epochs,
                               stats_interval,
                               notebook=True):

    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors = {'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    # Use the created objects to initialise a new Optimiser instance.
    optimiser = Optimiser(
        model,
        error,
        learning_rule,
        train_data,
        valid_data,
        test_data,
        data_monitors,
        notebook=notebook)

    # Run the optimiser for 5 epochs (full passes through the training set)
    # printing statistics every epoch.
    stats, keys, run_time = optimiser.train(
        num_epochs=num_epochs, stats_interval=stats_interval)

    return stats, keys, run_time

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

###### The model set up code below is provided as a starting point.
# You will probably want to add further code cells for the
# different experiments you run.

from mlp.layers import *
from mlp.errors import CrossEntropyLogSoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule, MomentumLearningRule
from mlp.optimisers import Optimiser

#setup hyperparameters
learning_rate = 0.01
num_epochs = 60 # TODO: CHANGED HERE FOR TESTING ONLY!
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 100


for trial in [2,3]:
    experiment_layers_BN_sigmoid = {}    
    for i in [1,3,5,7]:
        # reinitialisation of func needed for every expt!
        train_data.reset()
        test_data.reset()
        valid_data.reset()

        # Initialise the weights and biases:
        weights_init = GlorotUniformInit(rng=rng)
        biases_init = ConstantInit(0.)

        input_layer = [
            AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
            BatchNormalizationLayer(input_dim=hidden_dim, rng=rng)
        ]
        output_layer = [
            ReluLayer(),
            AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
        ]
        each_hidden_layer = [
            ReluLayer(),
            AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
            BatchNormalizationLayer(input_dim=hidden_dim, rng=rng),
        ]

        # create the MLP:
        model = MultipleLayerModel(input_layer + each_hidden_layer * i +
                                   output_layer)
        print(model, '{} layers'.format(i + 1))

        error = CrossEntropyLogSoftmaxError()
        learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

        experiment_layers_BN_sigmoid[i + 1] = train_model_and_plot_stats(
            model,
            error,
            learning_rule,
            train_data,
            valid_data,
            test_data,
            num_epochs,
            stats_interval,
            notebook=False)
        # Save the stuff:
    pkl.dump(experiment_layers_BN_sigmoid, open('./experiment_layers_BN_sigmoid.{}.pkl'.format(trial), 'wb'), protocol=-1)