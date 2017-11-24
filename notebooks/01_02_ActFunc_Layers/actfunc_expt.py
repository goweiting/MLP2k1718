## EXPERIMENTS FOR DIFFERENT NUMBER OF LAYERS FOR DIFFERENT ACTIVATION FUNCTIONS!
## THIS IS A HUGE EXPERIMENT, EXPECTED TO RUN FOR a DAY

from mlp.helper import *
import pickle as pkl
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
from mlp.layers import *
from mlp.errors import CrossEntropyLogSoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import *
from mlp.learning_rules import *
from mlp.optimisers import Optimiser

#setup hyperparameters
learning_rate = 0.01
num_epochs = 60 # TODO: CHANGED HERE FOR TESTING ONLY!
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 100

## EXPERIMENT PARAMETERS
trials = range(2) # 0,1
layers_depth = [1,3,5,7] # +1 to each for the total number of layers:

funcs = {'sigmoid':SigmoidLayer(), 
         'relu':ReluLayer(), 
         'elu':ELULayer(), 
         'selu':SELULayer()}


## BEGIN EXPERIMENT:
for trial in trials:
    for layer in layers_depth:
        for label,func in funcs.items():
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
            model = MultipleLayerModel(input_layer + each_hidden_layer * layer +
                                       output_layer)
            print(model, '{} layers'.format(layer + 1))

            error = CrossEntropyLogSoftmaxError()
            learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

            _ = train_model_and_plot_stats(
                model,
                error,
                learning_rule,
                train_data,
                valid_data,
                test_data,
                num_epochs,
                stats_interval)
            name = 'actfunc_expt_t{}_d{}_{}.pkl'.format(trial, layer, label)
            pkl.dump(_, open(name, 'wb'), protocol=-1)
            print(name, "PICKLED")