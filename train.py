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

from utils import *
from train_image_reader import *
from net import *

parser = argparse.ArgumentParser(description='')

parser.add_argument("--snapshot_dir", default='./snapshots', help="path of snapshots")
parser.add_argument("--image_size", type=int, default=256, help="load image size")
parser.add_argument("--x_data_txt_path", default='./datasets/x_traindata.txt', help="txt of x images")
parser.add_argument("--y_data_txt_path", default='./datasets/y_traindata.txt', help="txt of y images")
parser.add_argument("--random_seed", type=int, default=1234, help="random seed")
parser.add_argument('--base_lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=20, help='# of epoch to decay lr')
parser.add_argument("--lamda", type=float, default=10.0, help="L1 lamda")
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument("--summary_pred_every", type=int, default=200, help="times to summary.")
parser.add_argument("--save_pred_every", type=int, default=8000, help="times to save.")
parser.add_argument("--x_image_forpath", default='./datasets/train/X/images/', help="forpath of x training datas.")
parser.add_argument("--x_label_forpath", default='./datasets/train/X/labels/', help="forpath of x training labels.")
parser.add_argument("--y_image_forpath", default='./datasets/train/Y/images/', help="forpath of y training datas.")
parser.add_argument("--y_label_forpath", default='./datasets/train/Y/labels/', help="forpath of y training labels.")

args = parser.parse_args()

def save(saver, sess, logdir, step):
   model_name = 'model'
   checkpoint_path = os.path.join(logdir, model_name)
   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def get_data_lists(data_path):
    f = open(data_path, 'r')
    datas=[]
    for line in f:
        data = line.strip("\n")
        datas.append(data)
    return datas

def l1_loss(src, dst):
    return tf.reduce_mean(tf.abs(src - dst))

def gan_loss(src, dst):
    return tf.reduce_mean((src-dst)**2)

def main():
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    x_datalists = get_data_lists(args.x_data_txt_path) # a list of x images
    y_datalists = get_data_lists(args.y_data_txt_path) # a list of y images
    tf.set_random_seed(args.random_seed)
    x_img = tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size,3],name='x_img')
    x_label = tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size,3],name='x_label')
    y_img = tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size,3],name='y_img')
    y_label = tf.placeholder(tf.float32,shape=[1, args.image_size, args.image_size,3],name='y_label')

    fake_y = generator(image=x_img, reuse=False, name='generator_x2y') # G
    fake_x_ = generator(image=fake_y, reuse=False, name='generator_y2x') # S
    fake_x = generator(image=y_img, reuse=True, name='generator_y2x') # G'
    fake_y_ = generator(image=fake_x, reuse=True, name='generator_x2y') # S'

    dy_fake = discriminator(image=fake_y, gen_label = x_label, reuse=False, name='discriminator_y') # D
    dx_fake = discriminator(image=fake_x, gen_label = y_label, reuse=False, name='discriminator_x') # D'
    dy_real = discriminator(image=y_img, gen_label = y_label, reuse=True, name='discriminator_y') # D
    dx_real = discriminator(image=x_img, gen_label = x_label, reuse=True, name='discriminator_x') #D'

    final_loss = gan_loss(dy_fake, tf.ones_like(dy_fake)) + gan_loss(dx_fake, tf.ones_like(dx_fake)) + args.lamda*l1_loss(x_label, fake_x_) + args.lamda*l1_loss(y_label, fake_y_) # final objective function

    dy_loss_real = gan_loss(dy_real, tf.ones_like(dy_real))
    dy_loss_fake = gan_loss(dy_fake, tf.zeros_like(dy_fake))
    dy_loss = (dy_loss_real + dy_loss_fake) / 2

    dx_loss_real = gan_loss(dx_real, tf.ones_like(dx_real))
    dx_loss_fake = gan_loss(dx_fake, tf.zeros_like(dx_fake))
    dx_loss = (dx_loss_real + dx_loss_fake) / 2

    dis_loss = dy_loss + dx_loss # discriminator loss

    final_loss_sum = tf.summary.scalar("final_objective", final_loss)

    dx_loss_sum = tf.summary.scalar("dx_loss", dx_loss)
    dy_loss_sum = tf.summary.scalar("dy_loss", dy_loss)
    dis_loss_sum = tf.summary.scalar("dis_loss", dis_loss)
    discriminator_sum = tf.summary.merge([dx_loss_sum, dy_loss_sum, dis_loss_sum])

    x_images_summary = tf.py_func(cv_inv_proc, [x_img], tf.float32) #(1, 256, 256, 3) float32
    y_fake_cv2inv_images_summary = tf.py_func(cv_inv_proc, [fake_y], tf.float32) #(1, 256, 256, 3) float32
    x_label_summary = tf.py_func(label_proc, [x_label], tf.float32) #(1, 256, 256, 3) float32
    x_gen_label_summary = tf.py_func(label_inv_proc, [fake_x_], tf.float32) #(1, 256, 256, 3) float32
    image_summary = tf.summary.image('images', tf.concat(axis=2, values=[x_images_summary, y_fake_cv2inv_images_summary, x_label_summary, x_gen_label_summary]), max_outputs=3)

    summary_writer = tf.summary.FileWriter(args.snapshot_dir, graph=tf.get_default_graph())

    g_vars = [v for v in tf.trainable_variables() if 'generator' in v.name]
    d_vars = [v for v in tf.trainable_variables() if 'discriminator' in v.name]

    lr = tf.placeholder(tf.float32, None, name='learning_rate')
    d_optim = tf.train.AdamOptimizer(lr, beta1=args.beta1)
    g_optim = tf.train.AdamOptimizer(lr, beta1=args.beta1)

    d_grads_and_vars = d_optim.compute_gradients(dis_loss, var_list=d_vars)
    d_train = d_optim.apply_gradients(d_grads_and_vars) # update weights of D and D'
    g_grads_and_vars = g_optim.compute_gradients(final_loss, var_list=g_vars)
    g_train = g_optim.apply_gradients(g_grads_and_vars) # update weights of G, G', S and S'

    train_op = tf.group(d_train, g_train)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=50)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    counter = 0 # training step

    for epoch in range(args.epoch):
        shuffle(x_datalists) # change the order of x images
        shuffle(y_datalists) # change the order of y images
        lrate = args.base_lr if epoch < args.epoch_step else args.base_lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)
        for step in range(len(x_datalists)):
            counter += 1
            x_image_resize, x_label_resize, y_image_resize, y_label_resize = TrainImageReader(args.x_image_forpath, args.x_label_forpath, args.y_image_forpath, args.y_label_forpath, x_datalists, y_datalists, step, args.image_size)
            batch_x_image = np.expand_dims(np.array(x_image_resize).astype(np.float32), axis = 0)
            batch_x_label = np.expand_dims(np.array(x_label_resize).astype(np.float32), axis = 0)
            batch_y_image = np.expand_dims(np.array(y_image_resize).astype(np.float32), axis = 0)
            batch_y_label = np.expand_dims(np.array(y_label_resize).astype(np.float32), axis = 0)
            start_time = time.time()
            feed_dict = { lr : lrate, x_img : batch_x_image, x_label : batch_x_label, y_img : batch_y_image, y_label : batch_y_label}
            if counter % args.save_pred_every == 0:
                final_loss_value, dis_loss_value, _ = sess.run([final_loss, dis_loss, train_op], feed_dict=feed_dict)
                save(saver, sess, args.snapshot_dir, counter)
            elif counter % args.summary_pred_every == 0:
                final_loss_value, dis_loss_value, final_loss_sum_value, discriminator_sum_value, image_summary_value, _ = \
                    sess.run([final_loss, dis_loss, final_loss_sum, discriminator_sum, image_summary, train_op], feed_dict=feed_dict)
                summary_writer.add_summary(final_loss_sum_value, counter)
                summary_writer.add_summary(discriminator_sum_value, counter)
                summary_writer.add_summary(image_summary_value, counter)
            else:
                final_loss_value, dis_loss_value, _ = \
                    sess.run([final_loss, dis_loss, train_op], feed_dict=feed_dict)
            print('epoch {:d} step {:d} \t final_loss = {:.3f}, dis_loss = {:.3f}'.format(epoch, step, final_loss_value, dis_loss_value))
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
