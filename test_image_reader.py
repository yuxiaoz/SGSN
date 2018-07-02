import os

import numpy as np
import tensorflow as tf
import cv2

def TestImageReader(test_image_forpath, test_label_forpath, test_file_list, step, size):
    file_length = len(test_file_list)
    line_idx = step % file_length
    test_line_content = test_file_list[line_idx]
    test_image_name = test_line_content.split(' ')[0]
    test_label_name = test_line_content.split(' ')[1]
    test_image_path = test_image_forpath + test_image_name
    test_label_path = test_label_forpath + test_label_name
    test_image = cv2.imread(test_image_path,1)
    test_label = cv2.imread(test_label_path,1)
    test_image_resize_t = cv2.resize(test_image, (size, size))
    test_image_resize = test_image_resize_t/127.5-1 #proc to [-1,1]
    test_label_resize = cv2.resize(test_label, (size, size), interpolation=cv2.INTER_NEAREST)
    return test_image_name, test_label_name, test_image_resize, test_label_resize
