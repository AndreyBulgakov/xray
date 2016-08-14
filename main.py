import numpy as np
from scipy import ndimage, misc
from matplotlib import pyplot as plt
from utils.utils import init_input_by_name
import os
from utils.preprocessing import crop_lungs, Cropper
from utils.preprocessing import create_dataset, load_dataset
from models import alexnet, googlenet, inceptionresnet
from utils.utils import get_resized_image
# data_dir =  "01-Apr-2015/"
# file_list = os.listdir(data_dir)
# image_dir = "images/"
# num = 5
# print len(file_list)
# for file in file_list[num:num+1]:
#
#     image = init_input_by_name(file, folder=data_dir)
#     # plt.imsave(image_dir + file[:-4] + '_2.png', image)
#     cropper = Cropper()
#     image = crop_lungs(image, cropper)
#     original_size_res = get_resized_image(image, 2340)
#     plt.imsave(image_dir + file[:-4] + '2.png', original_size_res)
#     plt.imsave(image_dir + file[:-4] + '.png', image)


image_size = 1024
data_dir = 'data/'
create_dataset(image_size, data_dir)
X, Y, X_test, Y_test = load_dataset(data_dir)
network = alexnet.create_alexnet(image_size)
alexnet.train_alexnet(network, X, Y, X_test, Y_test)