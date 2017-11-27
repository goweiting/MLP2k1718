from __future__ import print_function
import numpy as np
import logging
from mlp.data_providers import EMNISTDataProvider
from mlp.layers import *
from mlp.errors import CrossEntropyLogSoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule, AdamLearningRule, RMSPropLearningRule
from mlp.optimisers import Optimiser, EarlyStoppingOptimiser
from mlp.penalty import *
from mlp.helper import *
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
  
func = ELULayer()
inits = {}
experiment = {}
learningRules = {
"RMS":RMSPropLearningRule(),
"SGD":GradientDescentLearningRule(learning_rate=learning_rate),
"ADAM":AdamLearningRule()
}


for trial in range(1,4):
    for learning_rate in [1e-3, 1e-2, 1e-1]:
        train_data.reset()
        test_data.reset()
        valid_data.reset()

        # Initialise the weights and biases:
        weights_init = GlorotUniformInit(rng=rng)
        biases_init = ConstantInit(0.)

        input_layer = [
            AffineLayer(input_dim, hidden_dim, weights_init, biases_init, L2Penalty(1e-5))
        ]
        output_layer = [
            func,
            AffineLayer(hidden_dim, output_dim, weights_init, biases_init, L2Penalty(1e-5))
        ]
        each_hidden_layer = [
            func,
            AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, L2Penalty(1e-5))
        ]

        # create the MLP:
        model = MultipleLayerModel(input_layer + each_hidden_layer * hidden_layers
                                   + output_layer)

        error = CrossEntropyLogSoftmaxError()
        learning_rule = AdamLearningRule(learning_rate=learning_rate)

        experiment[learning_rate] = train_model_and_plot_stats(
            model,
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
            steps=3, patience=5)
    
    
    #===================== pickle: =====================================#
    pkl.dump(experiment, open('LR_wReg_ADAM_{}.pkl'.format(trial),'wb'), protocol=-1)
