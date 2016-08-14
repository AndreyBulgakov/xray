import h5py
import tflearn
import utils
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

def create_alexnet(input_size):
    # Building 'AlexNet'
    network = input_data(shape=[None, input_size, input_size, 1])

    network = conv_2d(network, 48, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 128, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 192, 3, activation='relu')
    network = conv_2d(network, 192, 3, activation='relu')
    network = conv_2d(network, 128, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 2048, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2048, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network

def train_alexnet(network, X, Y, X_test, Y_test):
    # Training
    model = tflearn.DNN(network, checkpoint_path='alexnet_flug.ckpt',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='tensorboard_log/')
    model.fit(X, Y, n_epoch=1000, validation_set=(X_test, Y_test), shuffle=True,
              show_metric=True, batch_size=1, snapshot_step=80,
              snapshot_epoch=True, run_id='alexnet_flug')
