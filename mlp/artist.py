import matplotlib.pyplot as plt
import numpy as np

def plot_graph_mean_std(data, best_epochs):
    c = ['b','g','c','r','k','y','m']
    c_i=0
    fig1 = plt.figure(figsize=(5, 5))  # ERRROR
    fig2 = plt.figure(figsize=(5, 5))  # ERRROR
    ax11 = fig1.add_subplot(111)
    ax12 = fig2.add_subplot(111)
    fig3 = plt.figure(figsize=(5, 5))  # ACCURACY
    fig4 = plt.figure(figsize=(5, 5))  # ACCURACY
    ax21 = fig3.add_subplot(111)
    ax22 = fig4.add_subplot(111)
    for i in data.keys():
        # plot each data dataset's validation and train
        _data = data[i]
        # plot the accuracy for each layer
        _tr_acc = _data['train_acc']
        _tr_err = _data['train_err']
        _val_acc = _data['val_acc']
        _val_err = _data['val_err']
        _x = np.arange(0, len(_tr_err['mean']))
        ax11.set_xlabel('epoch')
        ax11.set_ylabel('training error')
        ax11.plot(_x, _tr_err['mean'], label=i, c=c[c_i])
        ax11.fill_between(
            _x,
            _tr_err['mean'] - _tr_err['std'],
            _tr_err['mean'] + _tr_err['std'],
            alpha=.3, color=c[c_i])
        ax12.set_ylabel('validation error')
        ax12.set_xlabel('epoch')
        ax12.plot(
            _x, _val_err['mean'], label=i, color=c[c_i])
        ax12.fill_between(
            _x,
            _val_err['mean'] - _val_err['std'],
            _val_err['mean'] + _val_err['std'],
            alpha=.3, color=c[c_i])

        ax21.set_xlabel('epoch')
        ax21.set_ylabel('training accuracy')
        ax21.plot(_x, _tr_acc['mean'], label=i, c=c[c_i])
        ax21.fill_between(
            _x,
            _tr_acc['mean'] - _tr_acc['std'],
            _tr_acc['mean'] + _tr_acc['std'],
            alpha=.3, color=c[c_i])
        ax22.set_ylabel('validation accuracy')
        ax22.set_xlabel('epoch')
        ax22.plot(_x, _val_acc['mean'], label=i, c=c[c_i])
        ax22.fill_between(
            _x,
            _val_acc['mean'] - _val_acc['std'],
            _val_acc['mean'] + _val_acc['std'],
            alpha=.3, color=c[c_i])
        
        ax11.set_title('TRAINING')
        ax12.set_title('VALIDATION')
        ax12.legend(loc=0)
        ax22.legend(loc=0)
        ax22.scatter(best_epochs[i]['idx'], best_epochs[i]['mean'], c=c[c_i], marker='v')
        c_i+=1
        
    return fig1, fig2, fig3, fig4

def plot_combined_layers(data):
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for (layer, output) in data.items():
        keys = output[1]
        stats = output[0]
        # plot the accuracy for each layer
        ax1.plot(
            np.arange(1, stats.shape[0]),
            stats[1:, keys['error(valid)']])
        ax2.plot(
            np.arange(1, stats.shape[0]),
            stats[1:, keys['acc(valid)']], 
            label=layer)

    ax2.legend(loc='best');
    ax1.set_ylabel('validation error')
    ax2.set_ylabel('validation accuracy')
    ax2.set_xlabel('epoch number')

def plot_param_histogram(param, fig_size=(6, 3), interval=[-1.5, 1.5]):
    """Plots a normalised histogram of an array of parameter values."""
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    ax.hist(param.flatten(), 50, interval, normed=True)
    ax.set_xlabel('Parameter value')
    ax.set_ylabel('Normalised frequency density')
    return fig, ax

def visualise_first_layer_weights(weights, fig_size=(5, 5)):
    """Plots a grid of first layer weights as feature maps."""
    fig = plt.figure(figsize=fig_size)
    num_feature_maps = weights.shape[0]
    grid_size = int(num_feature_maps**0.5)
    max_abs = np.abs(weights).max()
    tiled = -np.ones((30 * grid_size, 
                      30 * num_feature_maps // grid_size)) * max_abs
    for i, fm in enumerate(weights):
        r, c = i % grid_size, i // grid_size
        tiled[1 + r * 30:(r + 1) * 30 - 1, 
              1 + c * 30:(c + 1) * 30 - 1] = fm.reshape((28, 28))
    ax = fig.add_subplot(111)
    max_abs = np.abs(tiled).max()
    ax.imshow(tiled, cmap='Greys', vmin=-max_abs, vmax=max_abs)
    ax.axis('off')
    fig.tight_layout()
    plt.show()
    return fig, ax

def show_batch_of_images(img_batch, fig_size=(3, 3)):
    fig = plt.figure(figsize=fig_size)
    batch_size, im_height, im_width = img_batch.shape
    # calculate no. columns per grid row to give square grid
    grid_size = int(batch_size**0.5)
    # intialise empty array to tile image grid into
    tiled = np.empty((im_height * grid_size, 
                      im_width * batch_size // grid_size))
    # iterate over images in batch + indexes within batch
    for i, img in enumerate(img_batch):
        # calculate grid row and column indices
        r, c = i % grid_size, i // grid_size
        tiled[r * im_height:(r + 1) * im_height, 
              c * im_height:(c + 1) * im_height] = img
    ax = fig.add_subplot(111)
    ax.imshow(tiled, cmap='Greys') #, vmin=0., vmax=1.)
    ax.axis('off')
    fig.tight_layout()
    plt.show()
    return fig, ax


