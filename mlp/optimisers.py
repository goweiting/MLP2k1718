# -*- coding: utf-8 -*-
"""Model optimisers.

This module contains objects implementing (batched) stochastic gradient descent
based optimisation of models.
"""

import time
import logging
from collections import OrderedDict
import numpy as np
from mlp.layers import LayerWithParameters, StochasticLayerWithParameters

# import tqdm

logger = logging.getLogger(__name__)


class Optimiser(object):
    """Basic model optimiser."""

    def __init__(self, model, error, learning_rule, train_dataset,
                 valid_dataset=None, test_dataset=None,
                 data_monitors=None, notebook=False):
        """Create a new optimiser instance.

        Args:
            model: The model to optimise.
            error: The scalar error function to minimise.
            learning_rule: Gradient based learning rule to use to minimise
                error.
            train_dataset: Data provider for training set data batches.
            valid_dataset: Data provider for validation set data batches.
            data_monitors: Dictionary of functions evaluated on targets and
                model outputs (averaged across both full training and
                validation data sets) to monitor during training in addition
                to the error. Keys should correspond to a string label for
                the statistic being evaluated.
            notebook: A boolean indicating whether experiments are carried out
            in an ipython-notebook, useful for indicating which progress bar styles
            to use.
        """
        self.model = model
        self.error = error
        self.learning_rule = learning_rule
        self.learning_rule.initialise(self.model.params)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.data_monitors = OrderedDict([('error', error)])
        if data_monitors is not None:
            self.data_monitors.update(data_monitors)
        self.notebook = notebook
        # if notebook:
        #    self.tqdm_progress = tqdm.tqdm_notebook
        # else:
        #    self.tqdm_progress = tqdm.tqdm

    def do_training_epoch(self):
        """Do a single training epoch.

        This iterates through all batches in training dataset, for each
        calculating the gradient of the estimated error given the batch with
        respect to all the model parameters and then updates the model
        parameters according to the learning rule.
        """
        # with self.tqdm_progress(total=self.train_dataset.num_batches) as train_progress_bar:
        #     train_progress_bar.set_description("Epoch Progress")
        for inputs_batch, targets_batch in self.train_dataset:
            activations = self.model.fprop(inputs_batch)
            grads_wrt_outputs = self.error.grad(activations[-1], targets_batch)
            grads_wrt_params = self.model.grads_wrt_params(
                activations, grads_wrt_outputs)
            self.learning_rule.update_params(grads_wrt_params)
            # train_progress_bar.update(1)

    def eval_monitors(self, dataset, label):
        """Evaluates the monitors for the given dataset.

        Args:
            dataset: Dataset to perform evaluation with.
            label: Tag to add to end of monitor keys to identify dataset.

        Returns:
            OrderedDict of monitor values evaluated on dataset.
        """
        data_mon_vals = OrderedDict([(key + label, 0.) for key
                                     in self.data_monitors.keys()])
        for inputs_batch, targets_batch in dataset:
            activations = self.model.fprop(inputs_batch, evaluation=True)
            for key, data_monitor in self.data_monitors.items():
                data_mon_vals[key + label] += data_monitor(
                    activations[-1], targets_batch)
        for key, data_monitor in self.data_monitors.items():
            data_mon_vals[key + label] /= dataset.num_batches
        return data_mon_vals

    def get_epoch_stats(self):
        """Computes training statistics for an epoch.

        Returns:
            An OrderedDict with keys corresponding to the statistic labels and
            values corresponding to the value of the statistic.
        """
        epoch_stats = OrderedDict()
        epoch_stats.update(self.eval_monitors(self.train_dataset, '(train)'))
        if self.valid_dataset is not None:
            epoch_stats.update(self.eval_monitors(
                self.valid_dataset, '(valid)'))
        if self.test_dataset is not None:  # INCLUDE THE TEST STATISTICS!
            epoch_stats.update(self.eval_monitors(
                self.test_dataset, '(test)'))
        return epoch_stats

    def log_stats(self, epoch, epoch_time, stats):
        """Outputs stats for a training epoch to a logger.

        Args:
            epoch (int): Epoch counter.
            epoch_time: Time taken in seconds for the epoch to complete.
            stats: Monitored stats for the epoch.
        """
        logger.info('Epoch {0}: {1:.1f}s to complete\n    {2}'.format(
            epoch, epoch_time,
            ', '.join(['{0}={1:.2e}'.format(k, v) for (k, v) in stats.items()])
        ))

    def train(self, num_epochs, stats_interval=5):
        """Trains a model for a set number of epochs.

        Args:
            num_epochs: Number of epochs (complete passes through trainin
                dataset) to train for.
            stats_interval: Training statistics will be recorded and logged
                every `stats_interval` epochs.

        Returns:
            Tuple with first value being an array of training run statistics
            and the second being a dict mapping the labels for the statistics
            recorded to their column index in the array.
        """
        start_train_time = time.time()
        run_stats = [list(self.get_epoch_stats().values())]
        # with self.tqdm_progress(total=num_epochs) as progress_bar:
        #     progress_bar.set_description("Experiment Progress")
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            self.do_training_epoch()
            epoch_time = time.time() - start_time
            if epoch % stats_interval == 0:
                stats = self.get_epoch_stats()
                self.log_stats(epoch, epoch_time, stats)
                run_stats.append(list(stats.values()))
                # progress_bar.update(1)
        finish_train_time = time.time()
        total_train_time = finish_train_time - start_train_time
        return np.array(run_stats), {k: i for i, k in enumerate(stats.keys())}, total_train_time


class EarlyStoppingOptimiser(object):
    """
    Early Stopping Optimiser checks the training and validation accuracy every <patience> epoch,
    after the first <steps>*<patience> epochs had elapsed.
    The model must not be improving for <steps> of <patience> epoch before terminating.

    the smaller <patience> is, the more accumulative improvements the model must perform.
    a large <patience> allows for the model to experience some pertubations (such as sudden good trainig/validation)
    dataset before decreasing.
    """

    def __init__(self, model, error, learning_rule, train_dataset,
                 valid_dataset=None, test_dataset=None,
                 data_monitors=None, notebook=False, steps=3, patience=5):
        """Create a new optimiser instance.

        Args:
            model: The model to optimise.
            error: The scalar error function to minimise.
            learning_rule: Gradient based learning rule to use to minimise
                error.
            train_dataset: Data provider for training set data batches.
            valid_dataset: Data provider for validation set data batches.
            data_monitors: Dictionary of functions evaluated on targets and
                model outputs (averaged across both full training and
                validation data sets) to monitor during training in addition
                to the error. Keys should correspond to a string label for
                the statistic being evaluated.
            notebook: A boolean indicating whether experiments are carried out
            in an ipython-notebook, useful for indicating which progress bar styles
            to use.
            steps: successive strips to look into
            pateience: number of history to look into
        """
        self.model = model
        self.error = error
        self.learning_rule = learning_rule
        self.learning_rule.initialise(self.model.params)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.data_monitors = OrderedDict([('error', error)])
        if data_monitors is not None:
            self.data_monitors.update(data_monitors)
        self.notebook = notebook

        # FOR UPs generalisation error:
        self.steps = steps
        self.patience = patience

    def do_training_epoch(self):
        """Do a single training epoch.

        This iterates through all batches in training dataset, for each
        calculating the gradient of the estimated error given the batch with
        respect to all the model parameters and then updates the model
        parameters according to the learning rule.
        """
        for inputs_batch, targets_batch in self.train_dataset:
            activations = self.model.fprop(inputs_batch)
            grads_wrt_outputs = self.error.grad(activations[-1], targets_batch)

            grads_wrt_params = self.model.grads_wrt_params(
                activations, grads_wrt_outputs)
            self.learning_rule.update_params(grads_wrt_params)

    def eval_monitors(self, dataset, label):
        """Evaluates the monitors for the given dataset.

        Args:
            dataset: Dataset to perform evaluation with.
            label: Tag to add to end of monitor keys to identify dataset.
            e.g. `error` and `acc`

        Returns:
            OrderedDict of monitor values evaluated on dataset.
        """
        data_mon_vals = OrderedDict([(key + label, 0.) for key
                                     in self.data_monitors.keys()])

        for inputs_batch, targets_batch in dataset:

            # Forward Propagate to get the output from the model!
            activations = self.model.fprop(inputs_batch, evaluation=True)

            # append each of the statistics to the data that we are monitoring:
            for key, data_monitor in self.data_monitors.items():
                # data_monitors is the function that take in two prameters:
                # e.g. the CrossEntropLogSoftmaxError calculate the cross entropy for each classes using the
                # last layer's activation function:
                data_mon_vals[key + label] += data_monitor(activations[-1], targets_batch)

        for key, data_monitor in self.data_monitors.items():
            # Average the error/accuracy across batch!
            # it is assumed that the data is representative
            data_mon_vals[key + label] /= dataset.num_batches

        return data_mon_vals

    def get_epoch_stats(self):
        """Computes training statistics for an epoch.

        Returns:
            An OrderedDict with keys corresponding to the statistic labels and
            values corresponding to the value of the statistic.
        """
        epoch_stats = OrderedDict()
        epoch_stats.update(self.eval_monitors(self.train_dataset, '(train)'))
        if self.valid_dataset is not None:
            epoch_stats.update(self.eval_monitors(
                self.valid_dataset, '(valid)'))
        if self.test_dataset is not None:  # INCLUDE THE TEST STATISTICS!
            epoch_stats.update(self.eval_monitors(
                self.test_dataset, '(test)'))
        # STORE THE GRADIENTS:
        params_stats = {}
        for i, layer in enumerate(self.model.layers):
            # Go through each layer with parameters to get the
            if isinstance(layer, LayerWithParameters) or isinstance(layer, StochasticLayerWithParameters):
                params_stats[i] =layer.params
        return epoch_stats, params_stats

    def log_stats(self, epoch, epoch_time, stats):
        """Outputs stats for a training epoch to a logger.

        Args:
            epoch (int): Epoch counter.
            epoch_time: Time taken in seconds for the epoch to complete.
            stats: Monitored stats for the epoch.
        """
        logger.info('Epoch {0}: {1:.1f}s to complete\n    {2}'.format(
            epoch, epoch_time,
            ', '.join(['{0}={1:.2e}'.format(k, v) for (k, v) in stats.items()])
        ))

    def train(self, max_num_epochs, stats_interval=5):
        """Trains a model for a set number of epochs.

        Args:
            num_epochs: Number of epochs (complete passes through trainin
                dataset) to train for.
            stats_interval: Training statistics will be recorded and logged
                every `stats_interval` epochs.

        Returns:
            Tuple with first value being an array of training run statistics
            and the second being a dict mapping the labels for the statistics
            recorded to their column index in the array.
        """
        start_train_time = time.time()

        # zeroth epoch:
        epoch_stat, param_stat = self.get_epoch_stats()  # this will evaluate the model!
        e_val = [epoch_stat['error(valid)']]  # stores the validation error for each epoch
        run_stats = [list(epoch_stat.values())]  # THE FIRST EPOCH_STATS IS NEGLIGIBLE HERE
        param_stats = [param_stat]  # Store as an ordered Dict
        early_stop = False
        epoch = 0
        models = {}
        best_model, _epoch = None, None

        while not early_stop and epoch <= max_num_epochs:
            epoch += 1
            start_time = time.time()
            self.do_training_epoch()
            epoch_time = time.time() - start_time

            # DO EARLY STOPPING MONITORING:
            stats, params = self.get_epoch_stats()
            self.log_stats(epoch, epoch_time, stats)  # PRINT THE STATS
            e_val.append(stats['error(valid)'])

            if epoch > self.patience * self.steps:
                # Start checking UP from this epoch:
                models[epoch] = self.model  # Append for later storage of best model
                _epoch = epoch
                for i in range(self.steps):  # compare s successive strips of size <patience>
                    prev = _epoch - self.patience
                    if e_val[_epoch] > e_val[prev]:
                        logger.info(
                            'UP{}: error(valid) at {} = {:.2e} > at {} = {:.2e}'.format(i + 1, _epoch, e_val[_epoch],
                                                                                        prev, e_val[prev]))
                        _epoch = prev
                        if i == self.steps - 1:
                            logger.info('EARLY STOPPING')  # STOP!
                            early_stop = True
                            best_model = models[prev]  ## TO CHECK
                    else:
                        # No point checking since require all successive strips to satisfy the condition
                        break

            # Save the epoch stats:
            run_stats.append(list(stats.values()))
            param_stats.append(params)

        finish_train_time = time.time()
        total_train_time = finish_train_time - start_train_time

        # RETURN THE EARLY STOPPED EPOCH TOO:
        return np.array(run_stats), {k: i for i, k in
                                     enumerate(epoch_stat.keys())}, total_train_time, _epoch, best_model, param_stats
