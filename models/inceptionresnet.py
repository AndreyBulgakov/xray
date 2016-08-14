import h5py
import tflearn
import utils
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d, global_avg_pool
from tflearn.layers.core import input_data, dropout, fully_connected, activation
from tflearn.layers.merge_ops import merge, merge_outputs
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization, batch_normalization


def stem(incoming):
    incoming = conv_2d(incoming, 32, 3, strides=2, padding='valid')
    incoming = conv_2d(incoming, 32, 3, strides=1, padding='valid')
    incoming = conv_2d(incoming, 64, 3, strides=1, padding='same')

    left_branch = max_pool_2d(incoming, 3, strides=2, padding='valid')
    right_branch = conv_2d(incoming, 96, 3, strides=2, padding='valid')

    incoming = merge([left_branch, right_branch], mode='concat', axis=3)

    left_branch2 = conv_2d(incoming, 64, 1, strides=1, padding='same')
    left_branch2 = conv_2d(left_branch2, 96, 3, strides=1, padding='valid')

    right_branch2 = conv_2d(incoming, 64, 1, strides=1, padding='same')
    right_branch2 = conv_2d(right_branch2, 64, [7, 1], strides=1, padding='same')
    right_branch2 = conv_2d(right_branch2, 64, [1, 7], strides=1, padding='same')
    right_branch2 = conv_2d(right_branch2, 96, 3, strides=1, padding='valid')

    incoming = merge([left_branch2, right_branch2], mode='concat', axis=3)
    # print incoming[3]
    # TODO strides added
    left_branch3 = conv_2d(incoming, 192, 3, strides=2, padding='valid')
    # TODO kernel size?
    right_branch3 = max_pool_2d(incoming, 3, strides=2, padding='valid')
    incoming = merge([left_branch3, right_branch3], mode='concat', axis=3)
    return incoming


def inception_resA(incoming):
    """
    Strides = 1
    Padding = SAME
    :param incoming:
    :return:
    """
    column_1 = conv_2d(incoming, 32, 1, strides=1, padding='same')

    column_2 = conv_2d(incoming, 32, 1, strides=1, padding='same')
    column_2 = conv_2d(column_2, 32, 3, strides=1, padding='same')

    column_3 = conv_2d(incoming, 32, 1, strides=1, padding='same')
    column_3 = conv_2d(column_3, 48, 3, strides=1, padding='same')
    column_3 = conv_2d(column_3, 64, 3, strides=1, padding='same')

    concat = merge([column_1, column_2, column_3], mode='concat', axis=3)

    concat = conv_2d(concat, 384, 1, activation='linear')

    incoming = incoming + concat

    incoming = activation(incoming, activation='relu')

    incoming = batch_normalization(incoming)
    return incoming


def reduction_resA(incoming):
    column_1 = max_pool_2d(incoming, 3, strides=2, padding='valid')

    column_2 = conv_2d(incoming, 384, 3, strides=2, padding='valid')

    column_3 = conv_2d(incoming, 256, 1, strides=1, padding='same')
    column_3 = conv_2d(column_3, 256, 3, strides=1, padding='same')
    column_3 = conv_2d(column_3, 384, 3, strides=2, padding='valid')

    concat = merge([column_1, column_2, column_3], mode='concat', axis=3)

    concat = batch_normalization(concat)
    return concat


def inception_resB(incoming):
    """
    Strides = 1
    Padding = SAME
    :param incoming:
    :return:
    """
    column_1 = conv_2d(incoming, 192, 1, strides=1, padding='same')

    column_2 = conv_2d(incoming, 128, 1, strides=1, padding='same')
    column_2 = conv_2d(column_2, 160, [1, 7], strides=1, padding='same')
    column_2 = conv_2d(column_2, 192, [7, 1], strides=1, padding='same')

    concat = merge([column_1, column_2], mode='concat', axis=3)
    # TODO 1154 before
    concat = conv_2d(concat, 1152, 1, activation='linear')

    incoming = incoming + concat
    incoming = activation(incoming, activation='relu')

    incoming = batch_normalization(incoming)
    return incoming


def reduction_resB(incoming):
    column_1 = max_pool_2d(incoming, 3, strides=2, padding='valid')

    column_2 = conv_2d(incoming, 256, 1, strides=1, padding='same')
    column_2 = conv_2d(column_2, 384, 3, strides=2, padding='valid')

    column_3 = conv_2d(incoming, 256, 1, strides=1, padding='same')
    column_3 = conv_2d(column_3, 288, 3, strides=2, padding='valid')

    column_4 = conv_2d(incoming, 256, 1, strides=1, padding='same')
    column_4 = conv_2d(column_4, 288, 3, strides=1, padding='same')
    column_4 = conv_2d(column_4, 320, 3, strides=2, padding='valid')

    concat = merge([column_1, column_2, column_3, column_4], mode='concat', axis=3)

    concat = batch_normalization(concat)
    return concat


def inception_resC(incoming):
    """
    Strides = 1
    Padding = SAME
    :param incoming:
    :return:
    """
    column_1 = conv_2d(incoming, 192, 1, strides=1, padding='same')

    column_2 = conv_2d(incoming, 192, 1, strides=1, padding='same')
    column_2 = conv_2d(column_2, 224, [1, 7], strides=1, padding='same')
    column_2 = conv_2d(column_2, 256, [7, 1], strides=1, padding='same')

    concat = merge([column_1, column_2], mode='concat', axis=3)

    # TODO 2048 before
    concat = conv_2d(concat, 2144, 1, activation='linear')

    incoming = incoming + concat
    incoming = activation(incoming, activation='relu')

    incoming = batch_normalization(incoming)
    return incoming


def create_inception_resnet_v2(input_size):
    # Building 'Inception-Resnet-v2'
    network = input_data(shape=[None, input_size, input_size, 1])
    # Modules
    # Stem
    network = stem(network)
    network = activation(network, activation='relu')
    network = batch_normalization(network)
    # Inception A
    # network = batch_normalization(network)
    network = inception_resA(network)
    network = inception_resA(network)
    network = inception_resA(network)
    network = inception_resA(network)
    network = inception_resA(network)
    # Reduction A
    # network = batch_normalization(network)
    network = reduction_resA(network)
    network = activation(network, activation='relu')
    # Inception B
    # network = batch_normalization(network)
    network = inception_resB(network)
    network = inception_resB(network)
    network = inception_resB(network)
    network = inception_resB(network)
    network = inception_resB(network)
    network = inception_resB(network)
    network = inception_resB(network)
    network = inception_resB(network)
    network = inception_resB(network)
    network = inception_resB(network)
    # Reduction B
    # network = batch_normalization(network)
    network = reduction_resB(network)
    network = activation(network, activation='relu')
    # Inception C
    # network = batch_normalization(network)
    network = inception_resC(network)
    network = inception_resC(network)
    network = inception_resC(network)
    network = inception_resC(network)
    network = inception_resC(network)
    # Avg pool TODO size?
    # network = avg_pool_2d(network, 7, strides=2, padding='valid')
    network = global_avg_pool(network)
    network = dropout(network, 0.8)
    # Result
    network = fully_connected(network, 2, activation='softmax', weight_decay=0.9)
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.0045)

    return network

def train_resnet(network, X, Y, X_test, Y_test):
    # Training
    model = tflearn.DNN(network, checkpoint_path='checkpoints/inception_resnet_flug.ckpt',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='tensorboard_log/')
    model.fit(X, Y, n_epoch=1000, validation_set=(X_test, Y_test), shuffle=True,
              show_metric=True, batch_size=1, snapshot_step=80,
              snapshot_epoch=True, run_id='inception_resnet_flug')
