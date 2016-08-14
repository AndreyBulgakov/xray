# coding=utf-8
import os
import dicom
import numpy as np
import scipy.misc as scm
import random


data_dir = "01-Apr-2015/"
map_dir = "maps"
file_list = os.listdir(data_dir)
file_list.sort()

# TODO Сделать проверку через карту пикселей в полигоне

# file_name это путь к файлу после дата.дир
def init_map_by_name(file_name, image_size = 0):
    lungs_map = np.load((open(map_dir + file_name[-31:] + '.polys.map.npz', 'rb')))['map']
    if image_size > 0:
        lungs_map = scm.imresize(lungs_map, [image_size, image_size])
    lungs_map = lungs_map / 255.0
    return lungs_map

def init_input_by_name(file_name, image_size = 0, folder = data_dir):
    image = dicom.read_file(folder + file_name).pixel_array
    if image_size > 0:
        image = scm.imresize(image, [image_size, image_size])
    return image

def get_random_patch(shape, min_size = 256, max_size = None):
    #print("shape: %d, %d", shape[0], shape[1])
    if max_size == None:
        max_size = min(shape[0], shape[1])
    width = random.randrange(min_size, max_size + 1)
    height = random.randrange(max(min_size, int(width * 0.9)), min(max_size + 1, int(width * 1.1)))
    x = random.randrange(shape[0] - width + 1)
    y = random.randrange(shape[1] - height + 1)
    return x, y, width, height
    #return scm.imresize(pixel_array[x:x+width, y:y+height], [image_size, image_size])

def get_resized_image(pixel_array, image_size = 256):
    return scm.imresize(pixel_array, [image_size, image_size])