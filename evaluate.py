from __future__ import print_function

import argparse
from datetime import datetime
from random import shuffle
import os
import sys
import time
import math
import tensorflow as tf
import numpy as np
import cv2


from utils import *
from test_image_reader import *
from net import *

parser = argparse.ArgumentParser(description='')

parser.add_argument("--image_forpath", default='./datasets/test/X/images/', help="forpath of testing datas.")
parser.add_argument("--label_forpath", default='./datasets/test/X/labels/', help="forpath of testing labels.")
parser.add_argument("--data_txt_path", default='./datasets/x_testdata.txt', help="txt path of testing data.")
parser.add_argument("--image_size", type=int, default=256, help="load image size")
parser.add_argument("--num_steps", type=int, default=710, help="Number of training steps.")
parser.add_argument("--snapshots", default='./snapshots/',help="Path of Snapshots")
parser.add_argument("--output_path", default='./test/',help="Output Folder")

args = parser.parse_args()

def get_data_lists(data_path):
    f = open(data_path, 'r')
    datas=[]
    for line in f:
        data = line.strip("\n")
        datas.append(data)
    return datas

def main():
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    datalists = get_data_lists(args.data_txt_path)
    test_image = tf.placeholder(tf.float32,shape=[1, 256, 256, 3], name = 'test_image')
    test_label = tf.placeholder(tf.float32,shape=[1, 256, 256, 3], name = 'test_label')

    fake_y = generator(image=test_image, reuse=False, name='generator_x2y')
    s_raw = generator(image=fake_y, reuse=False, name='generator_y2x')
    s_res_raw = s_raw * 255.
    s_res = tf.clip_by_value(s_res_raw, 0., 255.)

    restore_var = [v for v in tf.global_variables() if 'generator' in v.name]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    saver = tf.train.Saver(var_list=restore_var,max_to_keep=1)
    checkpoint = tf.train.latest_checkpoint(args.snapshots)
    saver.restore(sess, checkpoint)

    for step in range(args.num_steps):
        img_name, label_name, image_resize, label_resize = TestImageReader(args.image_forpath, args.label_forpath, datalists, step, args.image_size)
        batch_image = np.expand_dims(np.array(image_resize).astype(np.float32), axis = 0)
        batch_label = np.expand_dims(np.array(label_resize).astype(np.float32), axis = 0)
        feed_dict = { test_image : batch_image, test_label : batch_label}

        s_img_t = sess.run(s_res, feed_dict = feed_dict)
        image_w = (image_resize + 1 )*127.5 # 3 channels BGR
        label_w_t = label_resize * 255. #[0,1]->[0,255] 3 channels
        label_w = (label_w_t - 255.) * (-1)
        domain_name = img_name.split('Sight')[0]
        domain_num = img_name.split('Sight')[1].split('.')[0]
        image_w_name = domain_name+domain_num+'_a'+'.png'
        label_w_name = domain_name+domain_num+'_b'+'.png'
        s_img_name = domain_name+domain_num+'_c'+'.png'
        cv2.imwrite(args.output_path+image_w_name, image_w)
        cv2.imwrite(args.output_path+label_w_name, label_w)
        s_img_tt = s_img_t.astype(np.float32)[0]
        s_img = (s_img_tt - 255.) * (-1)
        cv2.imwrite(args.output_path + s_img_name, s_img)
        print('step {:d}'.format(step))

if __name__ == '__main__':
    main()
