from __future__ import print_function
import logging
from mlp.data_providers import EMNISTDataProvider
from mlp.layers import *
from mlp.errors import CrossEntropyLogSoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule, AdamLearningRule, RMSPropLearningRule
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

# setup hyperparameters
learning_rate = 0.01
num_epochs = 80
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 47, 400

# ==================================================================================
func = ELULayer()
i = 4  # + 1 = 5 layers
learning_Rates = [.01, .05, .1, .2]
# ==================================================================================
for trial in [3]:
    experiment_BN = {}
    for learning_Rate in learning_Rates:
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
            ELULayer(),
            AffineLayer(hidden_dim, output_dim, weights_init, biases_init) # NO BN TOWARDS SOFTMAX
        ]
        each_hidden_layer = [
            ELULayer(),
            AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
            BatchNormalizationLayer(input_dim=hidden_dim, rng=rng) # BN AFTER OUTPUT == BN FOR INPUT TO NEXT LAYER
        ]

        # create the MLP:
        model = MultipleLayerModel(input_layer + each_hidden_layer * i +
                                   output_layer)
        error = CrossEntropyLogSoftmaxError()
        learning_rule = GradientDescentLearningRule(learning_rate=learning_Rate)

        experiment_BN[learning_Rate] = train_model_and_plot_stats(model,
                                                                   error,
                                                                   learning_rule,
                                                                   train_data,
                                                                   valid_data,
                                                                   test_data,
                                                                   num_epochs,
                                                                   stats_interval,
                                                                   notebook=False,
                                                                   displayGraphs=False)
    pkl.dump(experiment_BN, open('BN_LR_{}.pkl'.format(trial), 'wb'), protocol=-1)
