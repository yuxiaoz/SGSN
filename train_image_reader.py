import os

import numpy as np
import tensorflow as tf
import cv2

def TrainImageReader(x_image_forpath, x_label_forpath, y_image_forpath, y_label_forpath, x_file_list, y_file_list, step, size):
    file_length = len(x_file_list)
    line_idx = step % file_length
    x_line_content = x_file_list[line_idx]
    y_line_content = y_file_list[line_idx]
    x_image_name = x_line_content.split(' ')[0]
    x_label_name = x_line_content.split(' ')[1]
    y_image_name = y_line_content.split(' ')[0]
    y_label_name = y_line_content.split(' ')[1]
    x_image_path = x_image_forpath + x_image_name
    x_label_path = x_label_forpath + x_label_name
    y_image_path = y_image_forpath + y_image_name
    y_label_path = y_label_forpath + y_label_name
    x_image = cv2.imread(x_image_path,1)
    x_label = cv2.imread(x_label_path,1) #need a 3-channel label
    y_image = cv2.imread(y_image_path,1)
    y_label = cv2.imread(y_label_path,1) #need a 3-channel label
    x_image_resize_t = cv2.resize(x_image, (size, size))
    x_image_resize = x_image_resize_t/127.5-1. #proc to [-1,1]
    x_label_resize = cv2.resize(x_label, (size, size), interpolation=cv2.INTER_NEAREST)
    y_image_resize_t = cv2.resize(y_image, (size, size))
    y_image_resize = y_image_resize_t/127.5-1. #proc to [-1,1]
    y_label_resize = cv2.resize(y_label, (size, size), interpolation=cv2.INTER_NEAREST)
    return x_image_resize, x_label_resize, y_image_resize, y_label_resize
