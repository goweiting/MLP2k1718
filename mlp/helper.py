### HELPER FUNCTIONS
from mlp.errors import CrossEntropyLogSoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import *
from mlp.learning_rules import *
from mlp.optimisers import Optimiser,EarlyStoppingOptimiser
from scipy.stats import wilcoxon
from itertools import product

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def analyse_mean_std(data):
    """
    Find the best mean
    """
    best_epochs = {}
    for i in data.keys():
        # each parameter's validation and training result
        _data = data[i]
        # plot the accuracy for each layer
        _tr_acc = _data['train_acc']
        _tr_err = _data['train_err']
        _val_acc = _data['val_acc']
        _val_err = _data['val_err']
        _test_acc = _data['test_acc']
        best_epoch = np.argmax(_val_acc['mean'])
        best_epochs[i] = {'idx': best_epoch,
                       'mean': _val_acc['mean'][best_epoch],
                       'std':  _val_acc['std'][best_epoch],
                         'test_acc_mean': _test_acc['mean'][best_epoch],
                         'test_acc_std': _test_acc['std'][best_epoch]}
    return best_epochs

def generate_mean_std(stats):
    """
    stats ia dictionary of:
    keys : parameters being varied
    values: dataset
    """
    out = {}
    for param in stats.keys():
        # Iterate each parameter to generate stats for each and every epoch:
        data = stats[param]
        out[param] = {}
        for _param in data.keys():
            # this corresponds to the val_err/val_acc/train_acc/train_err:
            _data = data[_param] # This data is for each trial!
            _data_paded = numpy_fillna(_data)
            mean = np.nanmean(_data_paded, axis=0)
            std = np.nanstd(_data_paded, axis=0)
            out[param][_param] = {'mean': mean,
                                  'std' : std}
    return out

def numpy_fillna(data):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.empty(mask.shape)
    out[:] = np.nan
    out[mask] = np.concatenate(np.array(data))
    return out

def generatePairs(labels):
    n = len(labels)
    expLen = (n*(n-1)/2) # n choose 2
    l1 = [(a,b) for (a,b) in product(labels,labels) if a != b]
    l2 = []
    for (a,b) in l1:
        if (a,b) not in l2:
            l2.append((a,b))
        if (b,a) in l2:
            l2.remove((b,a))
    assert len(l2) == expLen
    return l2

def wilcoxonTest(stats):
    """
    Stats is a dictionary with keys as the labels, and  statistics for the variable used in the experiment
    Each label is a dictionary, with keys as: val_err, val_acc, train_err, train_acc, test_acc.
    
    Methodology:
    we do a Wilcoxon signed-rank test with p-value<0.001 to detect if performance of one parameter (label) 
    is better than the other
    """
    # generate tuple of labels:
    parameters = stats.keys()
    testPars = generatePairs(parameters)
    tVals = []
    
    # call wilcoxon test:
    for (a,b) in testPars:
        try:
            x = np.mean(stats[a]['val_acc'], axis=0)
            y = np.mean(stats[b]['val_acc'], axis=0)
        except ValueError:
            # The arrays do not have equal length - usually the case for early stopping
            # Naively slice the array:
            _a = stats[a]['val_acc']
            _b = stats[b]['val_acc']
            N_a = len(_a)
            M_a = int(min([len(_a[i]) for i in range(len(_a))]))
            N_b = len(_b)
            M_b = int(min([len(_b[i]) for i in range(len(_b))]))
            assert N_a == N_b
            M = min(M_a,M_b)
            N = N_a
            x, y = np.empty((N,M)),np.empty((N,M))
            for i in range(N):
                x[i] = _a[i][:M]*np.ones(M)
                y[i] = _b[i][:M]*np.ones(M)
            x.resize(N,M)
            y.resize(N,M)
            x = np.mean(x, axis=0) 
            y = np.mean(y, axis=0)
        except IndexError:
            x = stats[a]['val_acc']['mean']
            y = stats[b]['val_acc']['mean']
            
        t = wilcoxon(x,y)
        if t.pvalue >= 0.05:
            print(a,b,t)
        tVals.append((a,b,t))
    return tVals



def train_model_and_plot_stats(model,
                               error,
                               learning_rule,
                               train_data,
                               valid_data,
                               test_data,
                               num_epochs,
                               stats_interval,
                               notebook=False,
                               displayGraphs=False,
                               earlyStop=False,
                               steps=None,
                               patience=None):

    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors = {'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    if earlyStop:
        optimiser = EarlyStoppingOptimiser(model,
                                           error,
                                           learning_rule,
                                           train_data,
                                           valid_data,
                                           test_data,
                                           data_monitors,
                                           notebook=notebook,
                                           steps=steps, 
                                           patience=patience)
        
    else:
        # Use the created objects to initialise a new Optimiser instance.
        optimiser = Optimiser(model,
                              error,
                              learning_rule,
                              train_data,
                              valid_data,
                              test_data,
                              data_monitors,
                              notebook=notebook)
    
    # TRAINING
    output = optimiser.train(num_epochs, stats_interval)

    if displayGraphs:
        # Plot the change in the validation and training set error over training.
        fig_1 = plt.figure(figsize=(8, 4))
        ax_1 = fig_1.add_subplot(111)
        for k in ['error(train)', 'error(valid)']:
            ax_1.plot(
                np.arange(1, stats.shape[0]) * stats_interval,
                stats[1:, keys[k]],
                label=k)
        ax_1.legend(loc=0)
        ax_1.set_xlabel('Epoch number')

        # Plot the change in the validation and training set accuracy over training.
        fig_2 = plt.figure(figsize=(8, 4))
        ax_2 = fig_2.add_subplot(111)
        for k in ['acc(train)', 'acc(valid)']:
            ax_2.plot(
                np.arange(1, stats.shape[0]) * stats_interval,
                stats[1:, keys[k]],
                label=k)
        ax_2.legend(loc=0)
        ax_2.set_xlabel('Epoch number')
        
    return output
