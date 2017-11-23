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
input_dim, output_dim, hidden_dim = 784, 47, 500

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
        stats, keys, run_time, best_epoch = optimiser.train(
            max_num_epochs=num_epochs, stats_interval=stats_interval)
        return (stats, keys, run_time, best_epoch)
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
        return (stats, keys, run_time)

#==================================================================================
func = ELULayer()
experiment_layers_elu = {}
hidden_dim = 500
i = 9 # + 1 = 10 layers    
#==================================================================================

for incl_prob in [.25,.5,.75]:
    train_data.reset()
    test_data.reset()
    valid_data.reset()

    # Initialise the weights and biases:
    weights_init = GlorotUniformInit(rng=rng)
    biases_init = ConstantInit(0.)

    input_layer = [
        DropoutLayer(rng=rng, incl_prob=incl_prob, share_across_batch=True),
        AffineLayer(input_dim, hidden_dim, weights_init, biases_init)
    ]
    output_layer = [
        ELULayer(),
        DropoutLayer(rng=rng, incl_prob=incl_prob, share_across_batch=True),
        AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
    ]
    each_hidden_layer = [
        ELULayer(),
        DropoutLayer(rng=rng, incl_prob=incl_prob, share_across_batch=True),
        AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init)
    ]

    # create the MLP:
    model = MultipleLayerModel(input_layer + each_hidden_layer * i +
                               output_layer)
    print(model, '{} layers'.format(i + 1))

    error = CrossEntropyLogSoftmaxError()
    learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

    experiment_DROPOUT[incl_prob] = train_model_and_plot_stats(
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
    
    pkl.dump(experiment_DROPOUT, open('experiment_DROPOUT_NO_EARLYSTOP.pkl','wb'), protocol=-1)