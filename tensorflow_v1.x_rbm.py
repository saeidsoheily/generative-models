__author__ = 'Saeid SOHILY-KHAH'
"""
Generative Models: Tensorflow (MNIST) Restricted Boltzmann Machine (RBM) Implementation [using TensorFlow 1.x] 
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Load data
def load_data():
    '''
    Load MNIST dataset
    :return: mnist, img_size, num_classes:
    '''
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Get a sample of whole dataset [comment 3 lines to run code on whole mnist data)
    n_training, n_test = 8000, 2000
    X_train, y_train = X_train[:n_training], y_train[:n_training]
    X_test, y_test = X_test[:n_test], y_test[:n_test]

    # Reshaping
    img_size = X_train.shape[-1]
    X_train = X_train.reshape((-1, img_size * img_size))
    X_test = X_test.reshape((-1, img_size * img_size))

    # Normalization
    epsilon = 1e-6 # solve divison by zero
    X_train = (X_train - np.min(X_train, axis=0)) / (np.max(X_train, axis=0) + epsilon)  # normalization
    X_test = (X_test - np.min(X_train, axis=0)) / (np.max(X_train, axis=0) + epsilon)  # normalization

    return X_train, X_test, y_train, y_test, img_size


# Generate a batch of data for training model
def next_batch(batch_size, data, labels):
    '''
    Generate a batch by returnning batch_size of random data samples and labels
    :param batch_size:
    :param data:
    :param labels:
    :return:
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    batch_data = data[idx]
    batch_labels = labels[idx]
    return batch_data, batch_labels


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Load mnist data
    X_train, X_test, y_train, y_test, img_size = load_data()

    # RBM: Initialization
    n_visible_units = img_size * img_size # flatten image
    n_hidden_units = 128 # number of features in hidden layer
    learning_rate = 0.01
    batch_size = 64
    n_epochs = 2001

    # RBM: Graph variables, placeholders,...
    visible_bias = tf.placeholder("float", [n_visible_units]) # visible_bias is shared among all visible units
    hidden_bias = tf.placeholder("float", [n_hidden_units]) # hidden_bias is shared among all hidden units
    W = tf.placeholder("float", [n_visible_units, n_hidden_units]) # weights between neurons

    _visible_bias = np.zeros([n_visible_units])
    _hidden_bias = np.zeros([n_hidden_units])
    _W = np.zeros([n_visible_units, n_hidden_units])
    learned_visible_bias = np.zeros([n_visible_units])
    learned_hidden_bias = np.zeros([n_hidden_units])
    learned_W = np.zeros([n_visible_units, n_hidden_units])

    # RBM: Gibbs Sampling
    v0_state = tf.placeholder("float", [None, n_visible_units])

    h0_prob = tf.nn.sigmoid(tf.matmul(v0_state, W) + hidden_bias)  # probabilities of the hidden units
    h0_state = tf.nn.relu(tf.sign(h0_prob - tf.random_uniform(tf.shape(h0_prob))))  # sample h, given v

    v1_prob = tf.nn.sigmoid(tf.matmul(h0_state, tf.transpose(W)) + visible_bias)
    v1_state = tf.nn.relu(tf.sign(v1_prob - tf.random_uniform(tf.shape(v1_prob)))) # sample v, given h

    h1_prob = tf.nn.sigmoid(tf.matmul(v1_state, W) + hidden_bias)
    h1_state = tf.nn.relu(tf.sign(h1_prob - tf.random_uniform(tf.shape(h1_prob))))  # sample h, given v

    # RBM: Contrastive divergence
    W_Delta = tf.matmul(tf.transpose(v0_state), h0_prob) - tf.matmul(tf.transpose(v1_state), h1_prob)
    update_W = W + learning_rate * W_Delta
    update_visible_bias = visible_bias + learning_rate * tf.reduce_mean(v0_state - v1_state, 0)
    update_hidden_bias = hidden_bias + learning_rate * tf.reduce_mean(h0_state - h1_state, 0)

    # RBM: Define cost function and optimizer
    cost_function = tf.reduce_mean(tf.square(v0_state - v1_state))

    # TensorFlow: Create a session to run the defined tensorflow graph
    sess = tf.Session()  # create a session

    # TensorFlow: Initialize the variables
    init = tf.global_variables_initializer()

    sess.run(init) # execute the initializer

    # RBM: Training...
    loss_history = []
    for epoch in range(n_epochs):
        X_train_batch, y_train_batch = next_batch(batch_size, X_train, y_train)

        _feed_dict = {v0_state: X_train_batch,
                      visible_bias: _visible_bias,
                      hidden_bias: _hidden_bias,
                      W: _W}
        learned_feed_dict = {v0_state: X_train_batch,
                             visible_bias: learned_visible_bias,
                             hidden_bias: learned_hidden_bias,
                             W: learned_W}

        learned_W = sess.run(update_W, feed_dict=_feed_dict)
        learned_visible_bias = sess.run(update_visible_bias, feed_dict=_feed_dict)
        learned_hidden_bias = sess.run(update_hidden_bias, feed_dict=_feed_dict)

        _W, _visible_bias, _hidden_bias = learned_W, learned_visible_bias, learned_hidden_bias

        # Summarize result
        if epoch % 100 == 0:
            loss_history.append(sess.run(cost_function, feed_dict=learned_feed_dict))
            print('Epoch:{:<5}   ->     Reconstruction error (loss)={:.3f}'.format(epoch, round(loss_history[-1], 3)))

    # Plot settings
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(17, 9))  # set figure shape and size
    fig.tight_layout(pad=3.0)  # set the spacing between subplots in Matplotlib

    axes.plot(loss_history, label='Loss={:.3f}'.format(round(loss_history[-1], 3)), color='r')
    axes.set_xlim([0, (n_epochs//100) + 1])
    axes.set_xlabel('Epoch (1:100)')
    axes.set_ylabel('Loss')
    axes.set_title("RBM'S RECONSTRUCTION ERROR - LOSS (TRAINING)", fontsize=12)
    axes.legend(loc="upper right")

    # To save the plot locally
    plt.savefig('tensorflow_v1.x_rbm.png', bbox_inches='tight')
    plt.show()