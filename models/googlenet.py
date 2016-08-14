import h5py
import tflearn
import utils
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.merge_ops import merge, merge_outputs
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization


def inception3a(incoming):
    """
    Inception module.
    Activation: Linear
    Padding: Same
    :param incoming:
    :return: inception layer
    """
    # Columns
    column1x1 = conv_2d(incoming, 64, 1, strides=1)

    column3x3 = conv_2d(incoming, 96, 1, strides=1)
    column3x3 = conv_2d(column3x3, 128, 3, strides=1, )

    column5x5 = conv_2d(incoming, 16, 3, strides=1)
    column5x5 = conv_2d(column5x5, 32, 3, strides=1)

    column_pool = max_pool_2d(incoming, 3, strides=1)
    column_pool = conv_2d(column_pool, 32, 1, strides=1)

    result_layer = merge([column1x1, column3x3, column5x5, column_pool], mode='concat', axis=3)
    return result_layer


def inception3b(incoming):
    """
    Inception module.
    Activation: Linear
    Padding: Same
    :param incoming:
    :return: inception layer
    """
    # Columns
    column1x1 = conv_2d(incoming, 128, 1, strides=1)

    column3x3 = conv_2d(incoming, 128, 1, strides=1)
    column3x3 = conv_2d(column3x3, 192, 3, strides=1, )

    column5x5 = conv_2d(incoming, 32, 3, strides=1)
    column5x5 = conv_2d(column5x5, 96, 3, strides=1)

    column_pool = max_pool_2d(incoming, 3, strides=1)
    column_pool = conv_2d(column_pool, 64, 1, strides=1)

    result_layer = merge([column1x1, column3x3, column5x5, column_pool], mode='concat', axis=3)
    return result_layer


def inception4a(incoming):
    """
    Inception module.
    Activation: Linear
    Padding: Same
    :param incoming:
    :return: inception layer
    """
    # Columns
    column1x1 = conv_2d(incoming, 192, 1, strides=1)

    column3x3 = conv_2d(incoming, 96, 1, strides=1)
    column3x3 = conv_2d(column3x3, 208, 3, strides=1, )

    column5x5 = conv_2d(incoming, 16, 3, strides=1)
    column5x5 = conv_2d(column5x5, 48, 3, strides=1)

    column_pool = max_pool_2d(incoming, 3, strides=1)
    column_pool = conv_2d(column_pool, 64, 1, strides=1)

    result_layer = merge([column1x1, column3x3, column5x5, column_pool], mode='concat', axis=3)
    return result_layer


def inception4b(incoming):
    """
    Inception module.
    Activation: Linear
    Padding: Same
    :param incoming:
    :return: inception layer
    """
    # Columns
    column1x1 = conv_2d(incoming, 160, 1, strides=1)

    column3x3 = conv_2d(incoming, 112, 1, strides=1)
    column3x3 = conv_2d(column3x3, 224, 3, strides=1, )

    column5x5 = conv_2d(incoming, 24, 3, strides=1)
    column5x5 = conv_2d(column5x5, 64, 3, strides=1)

    column_pool = max_pool_2d(incoming, 3, strides=1)
    column_pool = conv_2d(column_pool, 64, 1, strides=1)

    result_layer = merge([column1x1, column3x3, column5x5, column_pool], mode='concat', axis=3)
    return result_layer


def inception4c(incoming):
    """
    Inception module.
    Activation: Linear
    Padding: Same
    :param incoming:
    :return: inception layer
    """
    # Columns
    column1x1 = conv_2d(incoming, 128, 1, strides=1)

    column3x3 = conv_2d(incoming, 128, 1, strides=1)
    column3x3 = conv_2d(column3x3, 256, 3, strides=1, )

    column5x5 = conv_2d(incoming, 24, 3, strides=1)
    column5x5 = conv_2d(column5x5, 64, 3, strides=1)

    column_pool = max_pool_2d(incoming, 3, strides=1)
    column_pool = conv_2d(column_pool, 64, 1, strides=1)

    result_layer = merge([column1x1, column3x3, column5x5, column_pool], mode='concat', axis=3)
    return result_layer


def inception4d(incoming):
    """
    Inception module.
    Activation: Linear
    Padding: Same
    :param incoming:
    :return: inception layer
    """
    # Columns
    column1x1 = conv_2d(incoming, 112, 1, strides=1)

    column3x3 = conv_2d(incoming, 144, 1, strides=1)
    column3x3 = conv_2d(column3x3, 288, 3, strides=1, )

    column5x5 = conv_2d(incoming, 32, 3, strides=1)
    column5x5 = conv_2d(column5x5, 64, 3, strides=1)

    column_pool = max_pool_2d(incoming, 3, strides=1)
    column_pool = conv_2d(column_pool, 64, 1, strides=1)

    result_layer = merge([column1x1, column3x3, column5x5, column_pool], mode='concat', axis=3)
    return result_layer


def inception4e(incoming):
    """
    Inception module.
    Activation: Linear
    Padding: Same
    :param incoming:
    :return: inception layer
    """
    # Columns
    column1x1 = conv_2d(incoming, 256, 1, strides=1)

    column3x3 = conv_2d(incoming, 160, 1, strides=1)
    column3x3 = conv_2d(column3x3, 320, 3, strides=1, )

    column5x5 = conv_2d(incoming, 32, 3, strides=1)
    column5x5 = conv_2d(column5x5, 128, 3, strides=1)

    column_pool = max_pool_2d(incoming, 3, strides=1)
    column_pool = conv_2d(column_pool, 128, 1, strides=1)

    result_layer = merge([column1x1, column3x3, column5x5, column_pool], mode='concat', axis=3)
    return result_layer


def inception5a(incoming):
    """
    Inception module.
    Activation: Linear
    Padding: Same
    :param incoming:
    :return: inception layer
    """
    # Columns
    column1x1 = conv_2d(incoming, 256, 1, strides=1)

    column3x3 = conv_2d(incoming, 160, 1, strides=1)
    column3x3 = conv_2d(column3x3, 320, 3, strides=1, )

    column5x5 = conv_2d(incoming, 32, 3, strides=1)
    column5x5 = conv_2d(column5x5, 128, 3, strides=1)

    column_pool = max_pool_2d(incoming, 3, strides=1)
    column_pool = conv_2d(column_pool, 128, 1, strides=1)

    result_layer = merge([column1x1, column3x3, column5x5, column_pool], mode='concat', axis=3)
    return result_layer


def inception5b(incoming):
    """
    Inception module.
    Activation: Linear
    Padding: Same
    :param incoming:
    :return: inception layer
    """
    # Columns
    column1x1 = conv_2d(incoming, 384, 1, strides=1)

    column3x3 = conv_2d(incoming, 192, 1, strides=1)
    column3x3 = conv_2d(column3x3, 384, 3, strides=1, )

    column5x5 = conv_2d(incoming, 48, 3, strides=1)
    column5x5 = conv_2d(column5x5, 128, 3, strides=1)

    column_pool = max_pool_2d(incoming, 3, strides=1)
    column_pool = conv_2d(column_pool, 128, 1, strides=1)

    result_layer = merge([column1x1, column3x3, column5x5, column_pool], mode='concat', axis=3)
    return result_layer


def create_googlenet(input_size):
    # Building 'GoogleNet'
    network = input_data(shape=[None, input_size, input_size, 1])

    network = conv_2d(network, 64, 7, strides=2, activation='linear')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 64, 1, strides=1, activation='linear', padding='valid')
    network = conv_2d(network, 192, 3, strides=1, activation='linear')
    network = local_response_normalization(network)
    network = max_pool_2d(network, 3, strides=2)
    # Inception
    network = inception3a(network)
    network = inception3b(network)
    network = max_pool_2d(network, 3, strides=2)
    network = inception4a(network)
    network = inception4b(network)
    network = inception4c(network)
    network = inception4d(network)
    network = inception4e(network)
    network = max_pool_2d(network, 3, strides=2)
    network = inception5a(network)
    network = inception5b(network)
    # Result
    network = avg_pool_2d(network, 7, strides=1, padding='valid')
    network = fully_connected(network, 2, activation='softmax', weight_decay=0.9)
    network = regression(network, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network

def train_googlenet(network, X, Y, X_test, Y_test):
    # Training
    model = tflearn.DNN(network, checkpoint_path='googlenet_flug.ckpt',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='tensorboard_log/')
    model.fit(X, Y, n_epoch=1000, validation_set=(X_test, Y_test), shuffle=True,
              show_metric=True, batch_size=2, snapshot_step=80,
              snapshot_epoch=True, run_id='googlenet_flug')
