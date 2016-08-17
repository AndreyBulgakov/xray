# coding=utf-8
# Run file to create polygons
# Захардкожен номер файла
import os
import dicom
import userconstants
import paint

data_dir = userconstants.data_dir + "01-Apr-2015/"
test_dir = userconstants.test_dir + "Lungs/"
file_list = os.listdir(data_dir)
file_list.sort()

# for f in file_list:
arr = dicom.read_file(data_dir + file_list[10]).pixel_array
arr[arr > 5000] = 5000
paint.show(file_list[10], arr)
# for f in file_list:
#     arr = dicom.read_file(data_dir + f).pixel_array
#     arr[arr > 5000] = 5000
#     paint.show(f, arr)
