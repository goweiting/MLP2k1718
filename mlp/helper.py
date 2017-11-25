### HELPER FUNCTIONS
from mlp.errors import CrossEntropyLogSoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import *
from mlp.learning_rules import *
from mlp.optimisers import Optimiser

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


def train_model_and_plot_stats(model,
                               error,
                               learning_rule,
                               train_data,
                               valid_data,
                               test_data,
                               num_epochs,
                               stats_interval,
                               notebook=False,
                               displayGraphs=False):

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
    stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

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

        return optimiser.model, stats, keys, run_time, fig_1, ax_1, fig_2, ax_2
    else:
        return optimiser.model, stats, keys, run_time
