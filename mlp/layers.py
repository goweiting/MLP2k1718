# -*- coding: utf-8 -*-
"""Layer definitions.

This module defines classes which encapsulate a single layer.batch

These layers map input activations to output activation with the `fprop`
method and map gradients with repsect to outputs to gradients with respect to
their inputs with the `bprop` method.

Some layers will have learnable parameters and so will additionally define
methods for getting and setting parameter and calculating gradients with
respect to the layer parameters.
"""

import numpy as np
import mlp.initialisers as init
from mlp import DEFAULT_SEED


class Layer(object):
    """Abstract class defining the interface for a layer."""

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        raise NotImplementedError()

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        raise NotImplementedError()


class LayerWithParameters(Layer):
    """Abstract class defining the interface for a layer with parameters."""

    def grads_wrt_params(self, inputs, grads_wrt_outputs):
        """Calculates gradients with respect to layer parameters.

        Args:
            inputs: Array of inputs to layer of shape (batch_size, input_dim).
            grads_wrt_to_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            List of arrays of gradients with respect to the layer parameters
            with parameter gradients appearing in same order in tuple as
            returned from `get_params` method.
        """
        raise NotImplementedError()

    def params_penalty(self):
        """Returns the parameter dependent penalty term for this layer.

        If no parameter-dependent penalty terms are set this returns zero.
        """
        raise NotImplementedError()

    @property
    def params(self):
        """Returns a list of parameters of layer.

        Returns:
            List of current parameter values. This list should be in the
            corresponding order to the `values` argument to `set_params`.
        """
        raise NotImplementedError()

    @params.setter
    def params(self, values):
        """Sets layer parameters from a list of values.

        Args:
            values: List of values to set parameters to. This list should be
                in the corresponding order to what is returned by `get_params`.
        """
        raise NotImplementedError()


class StochasticLayerWithParameters(Layer):
    """Specialised layer which uses a stochastic forward propagation."""

    def __init__(self, rng=None):
        """Constructs a new StochasticLayer object.

        Args:
            rng (RandomState): Seeded random number generator object.
        """
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

    def fprop(self, inputs, stochastic=True):
        """Forward propagates activations through the layer transformation.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            stochastic: Flag allowing different deterministic
                forward-propagation mode in addition to default stochastic
                forward-propagation e.g. for use at test time. If False
                a deterministic forward-propagation transformation
                corresponding to the expected output of the stochastic
                forward-propagation is applied.

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        raise NotImplementedError()

    def grads_wrt_params(self, inputs, grads_wrt_outputs):
        """Calculates gradients with respect to layer parameters.

        Args:
            inputs: Array of inputs to layer of shape (batch_size, input_dim).
            grads_wrt_to_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            List of arrays of gradients with respect to the layer parameters
            with parameter gradients appearing in same order in tuple as
            returned from `get_params` method.
        """
        raise NotImplementedError()

    def params_penalty(self):
        """Returns the parameter dependent penalty term for this layer.

        If no parameter-dependent penalty terms are set this returns zero.
        """
        raise NotImplementedError()

    @property
    def params(self):
        """Returns a list of parameters of layer.

        Returns:
            List of current parameter values. This list should be in the
            corresponding order to the `values` argument to `set_params`.
        """
        raise NotImplementedError()

    @params.setter
    def params(self, values):
        """Sets layer parameters from a list of values.

        Args:
            values: List of values to set parameters to. This list should be
                in the corresponding order to what is returned by `get_params`.
        """
        raise NotImplementedError()


class StochasticLayer(Layer):
    """Specialised layer which uses a stochastic forward propagation."""

    def __init__(self, rng=None):
        """Constructs a new StochasticLayer object.

        Args:
            rng (RandomState): Seeded random number generator object.
        """
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

    def fprop(self, inputs, stochastic=True):
        """Forward propagates activations through the layer transformation.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            stochastic: Flag allowing different deterministic
                forward-propagation mode in addition to default stochastic
                forward-propagation e.g. for use at test time. If False
                a deterministic forward-propagation transformation
                corresponding to the expected output of the stochastic
                forward-propagation is applied.

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        raise NotImplementedError()

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs. This should correspond to
        default stochastic forward-propagation.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        raise NotImplementedError()


class AffineLayer(LayerWithParameters):
    """Layer implementing an affine tranformation of its inputs.

    This layer is parameterised by a weight matrix and bias vector.
    """

    def __init__(self, input_dim, output_dim,
                 weights_initialiser=init.UniformInit(-0.1, 0.1),
                 biases_initialiser=init.ConstantInit(0.),
                 weights_penalty=None, biases_penalty=None):
        """Initialises a parameterised affine layer.

        Args:
            input_dim (int): Dimension of inputs to the layer.
            output_dim (int): Dimension of the layer outputs.
            weights_initialiser: Initialiser for the weight parameters.
            biases_initialiser: Initialiser for the bias parameters.
            weights_penalty: Weights-dependent penalty term (regulariser) or
                None if no regularisation is to be applied to the weights.
            biases_penalty: Biases-dependent penalty term (regulariser) or
                None if no regularisation is to be applied to the biases.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = weights_initialiser((self.output_dim, self.input_dim))
        self.biases = biases_initialiser(self.output_dim)
        self.weights_penalty = weights_penalty
        self.biases_penalty = biases_penalty

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        For inputs `x`, outputs `y`, weights `W` and biases `b` the layer
        corresponds to `y = W.dot(x) + b`.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        return self.weights.dot(inputs.T).T + self.biases

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        return grads_wrt_outputs.dot(self.weights)

    def grads_wrt_params(self, inputs, grads_wrt_outputs):
        """Calculates gradients with respect to layer parameters.

        Args:
            inputs: array of inputs to layer of shape (batch_size, input_dim)
            grads_wrt_to_outputs: array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim)

        Returns:
            list of arrays of gradients with respect to the layer parameters
            `[grads_wrt_weights, grads_wrt_biases]`.
        """

        grads_wrt_weights = np.dot(grads_wrt_outputs.T, inputs)
        grads_wrt_biases = np.sum(grads_wrt_outputs, axis=0)

        if self.weights_penalty is not None:
            grads_wrt_weights += self.weights_penalty.grad(self.weights)

        if self.biases_penalty is not None:
            grads_wrt_biases += self.biases_penalty.grad(self.biases)

        return [grads_wrt_weights, grads_wrt_biases]

    def params_penalty(self):
        """Returns the parameter dependent penalty term for this layer.

        If no parameter-dependent penalty terms are set this returns zero.
        """
        params_penalty = 0
        if self.weights_penalty is not None:
            params_penalty += self.weights_penalty(self.weights)
        if self.biases_penalty is not None:
            params_penalty += self.biases_penalty(self.biases)
        return params_penalty

    @property
    def params(self):
        """A list of layer parameter values: `[weights, biases]`."""
        return [self.weights, self.biases]

    @params.setter
    def params(self, values):
        self.weights = values[0]
        self.biases = values[1]

    def __repr__(self):
        return 'AffineLayer(input_dim={0}, output_dim={1})'.format(
            self.input_dim, self.output_dim)


class BatchNormalizationLayer(StochasticLayerWithParameters):
    """Layer implementing an affine tranformation of its inputs.

    This layer is parameterised by a weight matrix and bias vector.
    """

    def __init__(self, input_dim, rng=None):
        """Initialises a parameterised affine layer.
        Args:
            input_dim : Dimension of the input layer
        """
        super(BatchNormalizationLayer, self).__init__(rng)
        self.beta = np.random.normal(size=(input_dim))
        self.gamma = np.random.normal(size=(input_dim))
        self.epsilon = 0.00001
        self.cache = []  # Store parameters for each training minibatch
        self.input_dim = input_dim

    def fprop(self, inputs, stochastic=True):
        """Forward propagates inputs through a layer."""
        N, _ = inputs.shape
        if stochastic:  # TRAINING
            # calculate the mean for each batch. input is of shape (batch_size, input_dim)
            mu = 1. / N * np.sum(inputs, axis=0)  # Mean of each feature
            xmu = inputs - mu
            var = 1. / N * np.sum(xmu ** 2, axis=0)  # variance of each feature
            xhat = xmu * (1. / np.sqrt(var + self.epsilon))  # normalise inputs
            self.cache.append([mu, var])  # store mean and variance for inference

        else:  # INFERENCE!
            # using the population statistics instead:
            pop_mu, mu_var = np.mean(self.cache)
            pop_var = N / (N - 1.) * mu_var  # population variance
            xhat = (inputs - pop_mu) * (pop_var + self.epsilon) ** (-1. / 2.)

        # Same step for both:
        output = self.gamma * xhat + self.beta
        return output

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        # Adoped from : http://cthorey.github.io./backpropagation/

        N, _ = outputs.shape
        [mu, var] = self.cache[-1]  # the most recent one
        xmu = inputs - mu
        dh = (1. / N) * self.gamma * (var + self.epsilon) ** (-1. / 2.) * (
            N * grads_wrt_outputs - np.sum(grads_wrt_outputs, axis=0) - xmu * (var + self.epsilon) ** (-1.) * np.sum(
                grads_wrt_outputs * xmu, axis=0))
        return dh

    def grads_wrt_params(self, inputs, grads_wrt_outputs):
        """Calculates gradients with respect to layer parameters.

        Args:
            inputs: array of inputs to layer of shape (batch_size, input_dim)
            grads_wrt_to_outputs: array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim)

        Returns:
            list of arrays of gradients with respect to the layer parameters
            `[grads_wrt_gamma, grads_wrt_beta]`.
        """
        N, D = inputs.shape
        [mu, var] = self.cache[-1]
        xmu = inputs - mu
        dbeta = np.sum(grads_wrt_outputs, axis=0)
        dgamma = np.sum(xmu * (var + self.epsilon) ** (-1. / 2.) * grads_wrt_outputs, axis=0)
        return [dgamma, dbeta]

    def params_penalty(self):
        """Returns the parameter dependent penalty term for this layer.

        If no parameter-dependent penalty terms are set this returns zero.
        """
        params_penalty = 0

        return params_penalty

    @property
    def params(self):
        """A list of layer parameter values: `[gammas, betas]`."""
        return [self.gamma, self.beta]

    @params.setter
    def params(self, values):
        self.gamma = values[0]
        self.beta = values[1]

    def __repr__(self):
        return 'BatchNormalizationLayer(input_dim={0})'.format(
            self.input_dim)


class SigmoidLayer(Layer):
    """Layer implementing an element-wise logistic sigmoid transformation."""

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        For inputs `x` and outputs `y` this corresponds to
        `y = 1 / (1 + exp(-x))`.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        return 1. / (1. + np.exp(-inputs))

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        return grads_wrt_outputs * outputs * (1. - outputs)

    def __repr__(self):
        return 'SigmoidLayer'


class ConvolutionalLayer(LayerWithParameters):
    """Layer implementing a 2D convolution-based transformation of its inputs.
    The layer is parameterised by a set of 2D convolutional kernels, a four
    dimensional array of shape
        (num_output_channels, num_input_channels, kernel_dim_1, kernel_dim_2)
    and a bias vector, a one dimensional array of shape
        (num_output_channels,)
    i.e. one shared bias per output channel.
    Assuming no-padding is applied to the inputs so that outputs are only
    calculated for positions where the kernel filters fully overlap with the
    inputs, and that unit strides are used the outputs will have spatial extent
        output_dim_1 = input_dim_1 - kernel_dim_1 + 1
        output_dim_2 = input_dim_2 - kernel_dim_2 + 1
    """

    def __init__(self, num_input_channels, num_output_channels,
                 input_dim_1, input_dim_2,
                 kernel_dim_1, kernel_dim_2,
                 padding=0, stride=1,
                 kernels_init=init.UniformInit(-0.01, 0.01),
                 biases_init=init.ConstantInit(0.),
                 kernels_penalty=None, biases_penalty=None):
        """Initialises a parameterised convolutional layer.
        Args:
            num_input_channels (int): Number of channels in inputs to
                layer (this may be number of colour channels in the input
                images if used as the first layer in a model, or the
                number of output channels, a.k.a. feature maps, from a
                a previous convolutional layer).
            num_output_channels (int): Number of channels in outputs
                from the layer, a.k.a. number of feature maps.
            input_dim_1 (int): Size of first input dimension of each 2D
                channel of inputs.
            input_dim_2 (int): Size of second input dimension of each 2D
                channel of inputs.
            kernel_dim_1 (int): Size of first dimension of each 2D channel of
                kernels.
            kernel_dim_2 (int): Size of second dimension of each 2D channel of
                kernels.
            kernels_intialiser: Initialiser for the kernel parameters.
            biases_initialiser: Initialiser for the bias parameters.
            kernels_penalty: Kernel-dependent penalty term (regulariser) or
                None if no regularisation is to be applied to the kernels.
            biases_penalty: Biases-dependent penalty term (regulariser) or
                None if no regularisation is to be applied to the biases.
        """
        self.d0 = int(num_input_channels)
        self.d1 = int(num_output_channels)
        # INPUTS
        self.h0 = int(input_dim_1)
        self.w0 = int(input_dim_2)
        # KERNELS
        self.f1 = int(kernel_dim_1)
        self.f2 = int(kernel_dim_2)
        # OUTPUTS
        ## CHECK THE DIMENSIONS:
        assert (input_dim_1 + 2 * padding - kernel_dim_1) % stride == 0
        assert (input_dim_2 + 2 * padding - kernel_dim_2) % stride == 0
        self.h1 = int((input_dim_1 + 2 * padding - kernel_dim_1) * stride ** -1) + 1  # output_dim_1
        self.w1 = int((input_dim_2 + 2 * padding - kernel_dim_2) * stride ** -1) + 1  # output_dim_2

        self.padding = int(padding)
        self.stride = int(stride)
        self.kernels_init = kernels_init
        self.biases_init = biases_init
        self.kernels_shape = (self.d1, self.d0, self.f1, self.f2)
        self.inputs_shape = (None, self.d0, self.h0, self.w0)
        self.kernels = self.kernels_init(self.kernels_shape)
        self.biases = self.biases_init(self.d1)
        self.kernels_penalty = kernels_penalty
        self.biases_penalty = biases_penalty

        self.cache = None

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.
        For inputs `x`, outputs `y`, kernels `K` and biases `b` the layer
        corresponds to `y = conv2d(x, K) + b`.
        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        # convert inputs into columns for
        xCols = im2col_indices(inputs, field_height=self.f1, field_width=self.f2, padding=self.padding,
                               stride=self.stride)
        # Convert kernels into column vectors
        wCols = np.reshape(self.kernels, (self.d1, -1))
        # cross-correlation:
        out = wCols @ xCols + np.reshape(self.biases, (self.d1, -1))
        # reshape the output for each feature map:
        out = out.reshape(self.d1, self.h1, self.w1, inputs.shape[0])
        out = out.transpose(3, 0, 1, 2)

        self.cache = (xCols, out)
        return out

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.
        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.
        Args:
            inputs: Array of layer inputs of shape
                (batch_size, num_input_channels, input_dim_1, input_dim_2).
            outputs: Array of layer outputs calculated in forward pass of
                shape
                (batch_size, num_output_channels, output_dim_1, output_dim_2).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape
                (batch_size, num_output_channels, output_dim_1, output_dim_2).
        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        _grads_wrt_outputs = grads_wrt_outputs.transpose(1, 2, 3, 0).reshape(self.d1, -1)
        _kernels = self.kernels.reshape(self.d1, -1)
        _output = _kernels.T @ _grads_wrt_outputs
        output = col2im_indices(_output, inputs.shape, field_height=self.f1, field_width=self.f2, padding=self.padding,
                                stride=self.stride)
        assert output.shape == inputs.shape, "error transforming from column to matrices!"
        return output

    def grads_wrt_params(self, inputs, grads_wrt_outputs):
        """Calculates gradients with respect to layer parameters.
        Args:
            inputs: array of inputs to layer of shape (batch_size, input_dim)
            grads_wrt_to_outputs: array of gradients with respect to the layer
                outputs of shape
                (batch_size, num_output_channels, output_dim_1, output_dim_2).
        Returns:
            list of arrays of gradients with respect to the layer parameters
            `[grads_wrt_kernels, grads_wrt_biases]`.
        """
        (xCols, out) = self.cache
        _grads_wrt_biases = np.sum(grads_wrt_outputs, axis=(0, 2, 3))
        grads_wrt_biases = _grads_wrt_biases.reshape(self.d1, )

        _grads_wrt_outputs = grads_wrt_outputs.transpose(1, 2, 3, 0).reshape(self.d1, -1)
        _grads_wrt_kernels = _grads_wrt_outputs @ xCols.T
        grads_wrt_kernels = _grads_wrt_kernels.reshape(self.kernels_shape)

        return [grads_wrt_kernels, grads_wrt_biases]

    @property
    def params(self):
        """A list of layer parameter values: `[kernels, biases]`."""
        return [self.kernels, self.biases]

    @params.setter
    def params(self, values):
        self.kernels = values[0]
        self.biases = values[1]

    def __repr__(self):
        return (
            'ConvolutionalLayer(\n'
            '    num_input_channels={0}, num_output_channels={1},\n'
            '    input_dim_1={2}, input_dim_2={3},\n'
            '    kernel_dim_1={4}, kernel_dim_2={5}\n'
            ')'
                .format(self.d0, self.d1,
                        self.w0, self.h0, self.f1,
                        self.f2)
        )


class MaxPoolingLayer(Layer):
    """
    MaxPoolingLayer implements MaxPool for each input channel of the input. A receptive
    field of extent x extent is used to pool the areas. By default, no overlapping striding is used.
    i.e. a stride size = extent is used.
    """

    def __init__(self, num_input_channels, input_dim_1, input_dim_2, extent=2, stride=None):

        self.d0 = int(num_input_channels) # equivalent to num_output_channel
        # INPUTS
        self.h0 = int(input_dim_1)
        self.w0 = int(input_dim_2)
        # MAXPOOLING
        self.f = extent
        self.stride = extent if stride is None else stride
        # OUTPUTS
        assert (input_dim_1 - extent) % self.stride == 0
        assert (input_dim_2 - extent) % self.stride == 0
        self.h1 = int((input_dim_1 - extent) / self.stride + 1)
        self.w1 = int((input_dim_2 - extent) / self.stride + 1)

        self.cache = None

    def fprop(self, inputs):
        batch_size, _num_input_chn, _input_dim_1, _input_dim_2 = inputs.shape
        # Check that the input passes is similar ot what was declared
        assert _num_input_chn == self.d0
        assert _input_dim_1 == self.h0
        assert _input_dim_2 == self.w0

        # Stack all the minibatch together
        _inputs = inputs.reshape(batch_size * self.d0, 1, self.h0, self.w0)

        # cast the inputs into column vectors, each representing the area that the filter will meet
        # when a dot-product is used.
        xCols = im2col_indices(x=_inputs, field_height=self.f, field_width=self.f,
                               padding=0, stride=self.stride)

        # get the index of inputs that give us the maximum == maximum for each row:
        maxOut_idx = np.argmax(xCols, axis=0)  # this will be the mask that we will be using for backprop too
        maxOut = xCols[maxOut_idx, range(maxOut_idx.size)] # get the max values using the idx found
        self.cache = (xCols, maxOut_idx) # store

        #  Transform the output:
        output = maxOut.reshape(self.h1, self.w1, batch_size, self.d0)
        output = output.transpose(2, 3, 0, 1)
        return output


    def bprop(self, inputs, outputs, grads_wrt_outputs):
        batch_size, _num_input_chn, _input_dim_1, _input_dim_2 = grads_wrt_outputs.shape
        # check the output shape:
        assert _num_input_chn == self.d0 # the number of output channel is the same!
        assert _input_dim_1 == self.h1
        assert _input_dim_2 == self.w1

        xCols, maxOut_idx = self.cache
        dX_col = np.zeros_like(xCols)
        doutput_col = grads_wrt_outputs.transpose(2, 3, 0, 1).ravel() # flatten it

        # assign the mask with grads_wrt_output for positions where the maximum are seen
        dX_col[maxOut_idx, range(doutput_col.size)] = doutput_col
        dX = col2im_indices(dX_col, (batch_size * self.d0, 1, self.h0, self.w0),
                            self.f, self.f, padding=0, stride=self.stride)
        # reshape the gradients according to how it was:
        dX = dX.reshape((batch_size, self.d0, self.h0, self.w0))
        return dX

    def __repr__(self):
        return (
            'MaxPoolLayer(\n'
            '    num_input_channels={0},\n'
            '    input_dim_1={1}, input_dim_2={2},\n'
            '    extent={3}, stride={4}\n'
            ')'
                .format(self.d0,
                        self.h0, self.w0,
                        self.f, self.stride)
        )


class ReluLayer(Layer):
    """Layer implementing an element-wise rectified linear transformation."""

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        For inputs `x` and outputs `y` this corresponds to `y = max(0, x)`.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        return np.maximum(inputs, 0.)

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        return (outputs > 0) * grads_wrt_outputs

    def __repr__(self):
        return 'ReluLayer'


class LeakyReluLayer(Layer):
    """Layer implementing an element-wise rectified linear transformation."""

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        For inputs `x` and outputs `y` this corresponds to `y = max(0, x)`.
        """
        positive_inputs = np.maximum(inputs, 0.)

        negative_inputs = inputs
        negative_inputs[negative_inputs > 0] = 0.
        negative_inputs = negative_inputs * self.alpha

        outputs = positive_inputs + negative_inputs
        return outputs

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.
        """
        positive_gradients = (outputs > 0) * grads_wrt_outputs
        negative_gradients = self.alpha * (outputs < 0) * grads_wrt_outputs
        gradients = positive_gradients + negative_gradients
        return gradients

    def __repr__(self):
        return 'LeakyReluLayer'


class ELULayer(Layer):
    """Layer implementing an ELU activation."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        For inputs `x` and outputs `y` this corresponds to `y = max(0, x)`.
        """
        ## Clipping the inputs:
        # inputs = np.where(np.isnan(inputs), 0., inputs) # since they wont be assign to 0 anyway
        # inputs = np.where(np.isinf(inputs), 1e25, inputs)
        # positive_inputs = np.maximum(inputs, 0.)

        # negative_inputs = np.copy(inputs)
        ##negative_inputs[negative_inputs > 0.] = 0.
        # negative_inputs = np.where(negative_inputs > 0., 0., negative_inputs)
        # negative_inputs = self.alpha * (np.exp(negative_inputs) - 1.)

        # outputs = positive_inputs + negative_inputs
        ## Clipping the outputs:
        # outputs = np.where(np.isnan(outputs), -1e-25, outputs)
        # outputs = np.where(np.isinf(outputs), 1e25, outputs)
        # return outputs

        positive_inputs = np.maximum(inputs, 0.)

        negative_inputs = np.copy(inputs)
        negative_inputs[negative_inputs > 0] = 0.
        negative_inputs = self.alpha * (np.exp(negative_inputs) - 1)

        outputs = positive_inputs + negative_inputs
        return outputs

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.
        """
        ## Clipping:
        # grads_wrt_outputs = np.where(np.isnan(grads_wrt_outputs), -1e-25, grads_wrt_outputs)
        # grads_wrt_outputs = np.where(np.isinf(grads_wrt_outputs), 1e25, grads_wrt_outputs)

        ## positive_gradients = (outputs >= 0.) * grads_wrt_outputs
        # positive_gradients = np.where(outputs >= 0., 1. ,0.) * grads_wrt_outputs
        ## outputs_to_use = (outputs < 0.) * outputs
        # outputs_to_use = np.where(outputs < 0., 1., 0.) * outputs
        # negative_gradients = (outputs_to_use + self.alpha)
        ## negative_gradients[outputs >= 0.] = 0.
        # negative_gradients = np.where(outputs >= 0., 0., negative_gradients)
        # negative_gradients = negative_gradients * grads_wrt_outputs
        # gradients = positive_gradients + negative_gradients

        ## Gradient clipping:
        # gradients = np.where(np.isnan(gradients), -1e-25, gradients)
        # gradients = np.where(np.isinf(gradients), 1e25, gradients)


        positive_gradients = (outputs >= 0) * grads_wrt_outputs
        outputs_to_use = (outputs < 0) * outputs
        negative_gradients = (outputs_to_use + self.alpha)
        negative_gradients[outputs >= 0] = 0.
        negative_gradients = negative_gradients * grads_wrt_outputs
        gradients = positive_gradients + negative_gradients

        return gradients

    def __repr__(self):
        return 'ELULayer'


class SELULayer(Layer):
    """Layer implementing an element-wise rectified linear transformation."""

    # α01 ≈ 1.6733 and λ01 ≈ 1.0507
    def __init__(self):
        self.alpha = 1.6733
        self.lamda = 1.0507
        self.elu = ELULayer(alpha=self.alpha)

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        For inputs `x` and outputs `y` this corresponds to `y = max(0, x)`.
        """
        outputs = self.lamda * self.elu.fprop(inputs)
        return outputs

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.
        """
        scaled_outputs = outputs / self.lamda
        gradients = self.lamda * self.elu.bprop(inputs=inputs, outputs=scaled_outputs,
                                                grads_wrt_outputs=grads_wrt_outputs)
        return gradients

    def __repr__(self):
        return 'SELULayer'


class TanhLayer(Layer):
    """Layer implementing an element-wise hyperbolic tangent transformation."""

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        For inputs `x` and outputs `y` this corresponds to `y = tanh(x)`.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        return np.tanh(inputs)

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        return (1. - outputs ** 2) * grads_wrt_outputs

    def __repr__(self):
        return 'TanhLayer'


class SoftmaxLayer(Layer):
    """Layer implementing a softmax transformation."""

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        For inputs `x` and outputs `y` this corresponds to

            `y = exp(x) / sum(exp(x))`.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        # subtract max inside exponential to improve numerical stability -
        # when we divide through by sum this term cancels
        exp_inputs = np.exp(inputs - inputs.max(-1)[:, None])
        return exp_inputs / exp_inputs.sum(-1)[:, None]

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        return (outputs * (grads_wrt_outputs -
                           (grads_wrt_outputs * outputs).sum(-1)[:, None]))

    def __repr__(self):
        return 'SoftmaxLayer'


class RadialBasisFunctionLayer(Layer):
    """Layer implementing projection to a grid of radial basis functions."""

    def __init__(self, grid_dim, intervals=[[0., 1.]]):
        """Creates a radial basis function layer object.

        Args:
            grid_dim: Integer specifying how many basis function to use in
                grid across input space per dimension (so total number of
                basis functions will be grid_dim**input_dim)
            intervals: List of intervals (two element lists or tuples)
                specifying extents of axis-aligned region in input-space to
                tile basis functions in grid across. For example for a 2D input
                space spanning [0, 1] x [0, 1] use intervals=[[0, 1], [0, 1]].
        """
        num_basis = grid_dim ** len(intervals)
        self.centres = np.array(np.meshgrid(*[
            np.linspace(low, high, grid_dim) for (low, high) in intervals])
                                ).reshape((len(intervals), -1))
        self.scales = np.array([
            [(high - low) * 1. / grid_dim] for (low, high) in intervals])

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        return np.exp(-(inputs[..., None] - self.centres[None, ...]) ** 2 /
                      self.scales ** 2).reshape((inputs.shape[0], -1))

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        num_basis = self.centres.shape[1]
        return -2 * (
            ((inputs[..., None] - self.centres[None, ...]) / self.scales ** 2) *
            grads_wrt_outputs.reshape((inputs.shape[0], -1, num_basis))
        ).sum(-1)

    def __repr__(self):
        return 'RadialBasisFunctionLayer(grid_dim={0})'.format(self.grid_dim)


class DropoutLayer(StochasticLayer):
    """Layer which stochastically drops input dimensions in its output."""

    def __init__(self, rng=None, incl_prob=0.5, share_across_batch=True):
        """Construct a new dropout layer.

        Args:
            rng (RandomState): Seeded random number generator.
            incl_prob: Scalar value in (0, 1] specifying the probability of
                each input dimension being included in the output.
            share_across_batch: Whether to use same dropout mask across
                all inputs in a batch or use per input masks.
        """
        super(DropoutLayer, self).__init__(rng)
        assert incl_prob > 0. and incl_prob <= 1.
        self.incl_prob = incl_prob
        self.share_across_batch = share_across_batch
        self.rng = rng

    def fprop(self, inputs, stochastic=True):
        """Forward propagates activations through the layer transformation.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            stochastic: Flag allowing different deterministic
                forward-propagation mode in addition to default stochastic
                forward-propagation e.g. for use at test time. If False
                a deterministic forward-propagation transformation
                corresponding to the expected output of the stochastic
                forward-propagation is applied.

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        if stochastic:
            mask_shape = (1,) + inputs.shape[1:] if self.share_across_batch else inputs.shape
            self._mask = (self.rng.uniform(size=mask_shape) < self.incl_prob)
            return inputs * self._mask
        else:
            return inputs * self.incl_prob

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs. This should correspond to
        default stochastic forward-propagation.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        return grads_wrt_outputs * self._mask

    def __repr__(self):
        return 'DropoutLayer(incl_prob={0:.1f})'.format(self.incl_prob)


class ReshapeLayer(Layer):
    """Layer which reshapes dimensions of inputs."""

    def __init__(self, output_shape=None):
        """Create a new reshape layer object.

        Args:
            output_shape: Tuple specifying shape each input in batch should
                be reshaped to in outputs. This **excludes** the batch size
                so the shape of the final output array will be
                    (batch_size, ) + output_shape
                Similarly to numpy.reshape, one shape dimension can be -1. In
                this case, the value is inferred from the size of the input
                array and remaining dimensions. The shape specified must be
                compatible with the input array shape - i.e. the total number
                of values in the array cannot be changed. If set to `None` the
                output shape will be set to
                    (batch_size, -1)
                which will flatten all the inputs to vectors.
        """
        self.output_shape = (-1,) if output_shape is None else output_shape

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        return inputs.reshape((inputs.shape[0],) + self.output_shape)

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).
            outputs: Array of layer outputs calculated in forward pass of
                shape (batch_size, output_dim).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape (batch_size, output_dim).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        return grads_wrt_outputs.reshape(inputs.shape)

    def __repr__(self):
        return 'ReshapeLayer(output_shape={0})'.format(self.output_shape)


# =====================HELPER FUNCTIONS======================================#
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    else:
        return x_padded[:, :, padding:-padding, padding:-padding]
