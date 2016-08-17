import os
from scipy import ndimage
import skimage.transform
from skimage.transform import resize
import tensorflow as tf
import dicom
import numpy as np
import scipy.misc as scm
import utils
import matplotlib.pyplot as plt
from models.cropnet import create_cascade_net
import h5py


class Cropper(object):
    def __init__(self):
        batch_size = 1
        image_size = 256

        self.x = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 1])
        self.y = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 1])
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')
        self.loss, self.y_out = create_cascade_net(self.x, self.y, self.phase_train)
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, "checkpoints/cropnet.ckpt")

    def __del__(self):
        self.sess.close()

    def eval(self, batch, map_batch):
        with self.sess.as_default():
            res = self.y_out.eval(feed_dict={self.x: batch, self.y: map_batch, self.phase_train: False})
        return res


# def create_cropper():
#     # TODO make cropper an object
#     batch_size = 1
#     image_size = 256
#
#     x = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 1])
#     y = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 1])
#     phase_train = tf.placeholder(tf.bool, name='phase_train')
#
#     loss, y_out = create_cascade_net(x, y, phase_train)
#     sess = tf.InteractiveSession()
#     # sess = tf.Session()
#     saver = tf.train.Saver()
#     saver.restore(sess, "checkpoints/cropnet.ckpt")
#
#     return y_out, x, y, phase_train, sess


def crop_lungs(image, cropper):
    # orig = image
    batch_size = 1
    crop_image_size = 256
    image_size = image.shape[0]

    batch = np.ndarray(shape=(batch_size, crop_image_size, crop_image_size, 1), dtype='float32')
    map_batch = np.ndarray(shape=(batch_size, crop_image_size, crop_image_size, 1), dtype='float32')

    # Resize for croper
    batch[0, :, :, 0] = utils.get_resized_image(image, crop_image_size)

    res = cropper.eval(batch, map_batch)

    res_image = res[0, :, :, 0]

    # Filter for better bounds
    res_image = ndimage.median_filter(res_image, 15)

    # Delete not lungs
    res_image = res_image == 1

    # Two resizes???
    res_image = resize(res_image, (image_size, image_size))
    # res_image = utils.get_resized_image(res_image, image_size)
    # res_image = res_image != 0

    # Delete not lungs from original image
    image = image * res_image

    # Crop bounds
    res_x, res_y = np.where(image != 0)
    res_x = np.sort(res_x)
    res_y = np.sort(res_y)
    x1 = res_x[0]
    y1 = res_y[0]
    x2 = res_x[-1]
    y2 = res_y[-1]

    image = image[x1:x2, y1:y2]

    # plt.imsave('images/test.png', image, cmap='Greys')
    # image = orig[x1:x2, y1:y2]
    # plt.imsave('images/test2.png', image, cmap='Greys')
    return image


def collect_lungs(dir, image_size, cropper, patsize=-1, nonpatsize=-1):
    pat_list = os.listdir(dir + 'pat/')
    pat_list_size = len(pat_list)
    pat_list.sort()

    nonpat_list = os.listdir(dir + 'nonpat/')
    nonpat_list_size = len(nonpat_list)
    nonpat_list.sort()

    # Create cropper
    # cropper = Cropper()

    if patsize > pat_list_size or patsize == -1:
        # print '[WARNING] Count of patalogy is lower then you want or patsize == -1'
        patsize = pat_list_size
    if nonpatsize > nonpat_list_size or nonpatsize == -1:
        # print '[WARNING] Count of nonpatalogy is lower then you want or nonpatsize == -1'
        nonpatsize = nonpat_list_size

    batch_size = patsize + nonpatsize

    batch = np.ndarray(shape=(batch_size, image_size, image_size, 1), dtype='float32')
    labels = np.ndarray(shape=(batch_size, 2), dtype='int32')
    for i in range(patsize):
        file_name = pat_list[i]
        image = dicom.read_file(dir + 'pat/' + file_name).pixel_array
        image = crop_lungs(image, cropper)
        image = resize(image, (image_size, image_size))
        batch[i, :, :, 0] = image
        # batch[i, :, :, 0] = scm.imresize(dicom.read_file(dir + 'pat/' +file_name).pixel_array, [image_size, image_size]) \
        #                     / 255.0
        # batch[i, :, :, 0] = dicom.read_file(dir + 'pat/' +file_name).pixel_array / 255.0
        labels[i, :] = [1, 0]
    for i in range(nonpatsize):
        file_name = nonpat_list[i]
        image = dicom.read_file(dir + 'nonpat/' + file_name).pixel_array
        image = crop_lungs(image, cropper)
        image = resize(image, (image_size, image_size))
        batch[i + patsize, :, :, 0] = image
        # batch[i+patsize, :, :, 0] = scm.imresize(dicom.read_file(dir + 'nonpat/' +file_name).pixel_array, [image_size, image_size]) \
        #                             / 255.0
        # batch[i+patsize, :, :, 0] = dicom.read_file(dir + 'nonpat/'+ file_name).pixel_array / 255.0
        labels[i + patsize, :] = [0, 1]
    return batch, labels


def collect_batch(image_size, data_dir):
    train_dir = data_dir + 'train/'
    test_dir = data_dir + 'test/'

    cropper = Cropper()

    print "Colecting train lungs"
    trainX, trainY = collect_lungs(train_dir, image_size, cropper)

    print "Colecting test lungs"
    testX, testY = collect_lungs(test_dir, image_size, cropper)

    del cropper

    return trainX, trainY, testX, testY


def create_dataset(image_size, data_dir, ):
    print "Creating dataset"
    X, Y, testX, testY = collect_batch(image_size, data_dir)

    # Reshape for tensorflow???
    # X = X.reshape([-1, image_size, image_size, 1])
    # testX = testX.reshape([-1, image_size, image_size, 1])

    # Create dataset
    h5f = h5py.File(data_dir + 'full_data_set.h5', 'w')
    h5f.create_dataset('X', data=X)
    h5f.create_dataset('Y', data=Y)
    h5f.create_dataset('testX', data=testX)
    h5f.create_dataset('testY', data=testY)
    h5f.close()


def load_dataset(data_dir):
    # Get dataset
    h5f = h5py.File(data_dir + 'full_data_set.h5', 'r')
    X = h5f['X']
    Y = h5f['Y']
    X_test = h5f['testX']
    Y_test = h5f['testY']
    return X, Y, X_test, Y_test
