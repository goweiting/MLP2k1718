from __future__ import print_function
import numpy as np
import logging
from mlp.data_providers import EMNISTDataProvider
from mlp.layers import *
from mlp.errors import CrossEntropyLogSoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import *
from mlp.learning_rules import GradientDescentLearningRule, AdamLearningRule, RMSPropLearningRule
from mlp.optimisers import Optimiser, EarlyStoppingOptimiser
from mlp.helper import train_model_and_plot_stats
import pickle as pkl

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

#setup hyperparameters
learning_rate = 0.01
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 400 ### FOUR HUNDRED HIDDEN UNITS
hidden_layers = 3
  
## INSPECTION OF HIDDEN UNITS WITH ELU:
func = ELULayer()
inits = {}


for trial in range(1,4):
    experiment_units_elu = {}
    train_data.reset()
    test_data.reset()
    valid_data.reset()

    # Initialise the weights and biases:
    weights_init = GlorotUniformInit(rng=rng)
    biases_init = ConstantInit(0.)

    input_layer = [
        AffineLayer(input_dim, hidden_dim, weights_init, biases_init)
    ]
    output_layer = [
        func,
        AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
    ]
    each_hidden_layer = [
        func,
        AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init)
    ]

    # create the MLP:
    model = MultipleLayerModel(input_layer + each_hidden_layer * hidden_layers
                               + output_layer)

    error = CrossEntropyLogSoftmaxError()
    learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

    experiment_units_elu['glorot'] = train_model_and_plot_stats(
        model,
        error,
        learning_rule,
        train_data,
        valid_data,
        test_data,
        num_epochs,
        stats_interval,
        notebook=False,
        displayGraphs=False)
    
    
    #### ==================================================================================

    train_data.reset()
    test_data.reset()
    valid_data.reset()

    # Initialise the weights and biases:
    biases_init = ConstantInit(0.)

    input_layer = [
        AffineLayer(input_dim, hidden_dim, HeNormalInit(fan_in=input_dim, rng=rng), biases_init)
    ]
    output_layer = [
        func,
        AffineLayer(hidden_dim, output_dim, HeNormalInit(fan_in=hidden_dim, rng=rng), biases_init)
    ]
    each_hidden_layer = [
        func,
        AffineLayer(hidden_dim, hidden_dim, HeNormalInit(fan_in=hidden_dim, rng=rng), biases_init)
    ]

    # create the MLP:
    model = MultipleLayerModel(input_layer + each_hidden_layer * hidden_layers
                               + output_layer)

    error = CrossEntropyLogSoftmaxError()
    learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

    experiment_units_elu['he'] = train_model_and_plot_stats(
        model,
        error,
        learning_rule,
        train_data,
        valid_data,
        test_data,
        num_epochs,
        stats_interval,
        notebook=False,
        displayGraphs=False)
    
    
    #===================== pickle: =====================================#
    pkl.dump(experiment_units_elu, open('he_vs_glorot_{}.pkl'.format(trial),'wb'), protocol=-1)
