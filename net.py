import numpy as np
import tensorflow as tf
import math

def make_var(name, shape, trainable = True):
    return tf.get_variable(name, shape, trainable = trainable)

def conv2d(input_, output_dim, kernel_size, stride, padding = "SAME", name = "conv2d", biased = False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape=[kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.conv2d(input_, kernel, [1, stride, stride, 1], padding = padding)
        if biased:
            biases = make_var(name = 'biases', shape = [output_dim])
            output = tf.nn.bias_add(output, biases)
        return output

def atrous_conv2d(input_, output_dim, kernel_size, dilation, padding = "SAME", name = "atrous_conv2d", biased = False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape = [kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.atrous_conv2d(input_, kernel, dilation, padding = padding)
        if biased:
            biases = make_var(name = 'biases', shape = [output_dim])
            output = tf.nn.bias_add(output, biases)
        return output

def deconv2d(input_, output_dim, kernel_size, stride, padding = "SAME", name = "deconv2d"):
    input_dim = input_.get_shape()[-1]
    input_height = int(input_.get_shape()[1])
    input_width = int(input_.get_shape()[2])
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape = [kernel_size, kernel_size, output_dim, input_dim])
        output = tf.nn.conv2d_transpose(input_, kernel, [1, input_height * 2, input_width * 2, output_dim], [1, 2, 2, 1], padding = "SAME")
        return output

def batch_norm(input_, name="batch_norm"):
    with tf.variable_scope(name):
        input_dim = input_.get_shape()[-1]
        scale = tf.get_variable("scale", [input_dim], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [input_dim], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input_, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input_-mean)*inv
        output = scale*normalized + offset
        return output

def max_pooling(input_, kernel_size, stride, name, padding = "SAME"):
    return tf.nn.max_pool(input_, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding, name=name)

def avg_pooling(input_, kernel_size, stride, name, padding = "SAME"):
    return tf.nn.avg_pool(input_, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding, name=name)

def pyramid_pooling(input_, output_dim, kernel_size, stride = 1, padding = "SAME", name = "pyramid"):
    input_dim = input_.get_shape()[-1]
    avg_out = avg_pooling(input_ = input_, kernel_size = kernel_size, stride = kernel_size, name = name + "pa")
    conv1 = conv2d(input_ = avg_out, output_dim = output_dim, kernel_size = 1, stride = 1,  padding = "SAME", name = name + "py_conv2d", biased = False)
    output = tf.image.resize_bilinear(conv1, tf.shape(input_)[1:3,])
    return output

def lrelu(x, leak=0.2, name = "lrelu"):
    return tf.maximum(x, leak*x)

def relu(input_, name = "relu"):
    return tf.nn.relu(input_, name = name)

def residule_block_131(input_, output_dim1, output_dim2, stride = 1, dilation = 2, atrous = True, name = "res"):
    conv2dc0 = conv2d(input_ = input_, output_dim = output_dim1, kernel_size = 1, stride = stride, name = (name + '_c0'))
    conv2dc0_norm = batch_norm(input_ = conv2dc0, name = (name + '_bn0'))
    conv2dc1 = conv2d(input_ = input_, output_dim = output_dim2, kernel_size = 1, stride = stride, name = (name + '_c1'))
    conv2dc1_norm = batch_norm(input_ = conv2dc1, name = (name + '_bn1'))
    conv2dc1_relu = relu(input_ = conv2dc1_norm)
    if atrous:
        conv2dc2 = atrous_conv2d(input_ = conv2dc1_relu, output_dim = output_dim2, kernel_size = 3, dilation = dilation, name = (name + '_c2'))
    else:
        conv2dc2 = conv2d(input_ = conv2dc1_relu, output_dim = output_dim2, kernel_size = 3, stride = stride, name = (name + '_c2'))
    conv2dc2_norm = batch_norm(input_ = conv2dc2, name = (name + '_bn2'))
    conv2dc2_relu = relu(input_ = conv2dc2_norm)
    conv2dc3 = conv2d(input_ = conv2dc2_relu, output_dim = output_dim1, kernel_size = 1, stride = stride, name = (name + '_c3'))
    conv2dc3_norm = batch_norm(input_ = conv2dc3, name = (name + '_bn3'))
    add_raw = conv2dc0_norm + conv2dc3_norm
    output = relu(input_ = add_raw)
    return output

def residule_block_33(input_, output_dim, kernel_size = 3, stride = 1, dilation = 2, atrous = False, name = "res"):
    if atrous:
        conv2dc0 = atrous_conv2d(input_ = input_, output_dim = output_dim, kernel_size = kernel_size, dilation = dilation, name = (name + '_c0'))
        conv2dc0_norm = batch_norm(input_ = conv2dc0, name = (name + '_bn0'))
        conv2dc0_relu = relu(input_ = conv2dc0_norm)
        conv2dc1 = atrous_conv2d(input_ = conv2dc0_relu, output_dim = output_dim, kernel_size = kernel_size, dilation = dilation, name = (name + '_c1'))
        conv2dc1_norm = batch_norm(input_ = conv2dc1, name = (name + '_bn1'))
    else:
        conv2dc0 = conv2d(input_ = input_, output_dim = output_dim, kernel_size = kernel_size, stride = stride, name = (name + '_c0'))
        conv2dc0_norm = batch_norm(input_ = conv2dc0, name = (name + '_bn0'))
        conv2dc0_relu = relu(input_ = conv2dc0_norm)
        conv2dc1 = conv2d(input_ = conv2dc0_relu, output_dim = output_dim, kernel_size = kernel_size, stride = stride, name = (name + '_c1'))
        conv2dc1_norm = batch_norm(input_ = conv2dc1, name = (name + '_bn1'))
    add_raw = input_ + conv2dc1_norm
    output = relu(input_ = add_raw)
    return output

def generator(image, gf_dim=64, reuse=False, name="generator"): 
    #input_ : 1*256*256*3  
    input_dim = image.get_shape()[-1]
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        #c0_output: : 1*256*256*64  
        c0 = relu(batch_norm(conv2d(input_ = image, output_dim = gf_dim, kernel_size = 3, stride = 1, name = 'g_e0_c'), name = 'g_e0_bn'))
        #c1_output: : 1*128*128*128
        c1 = relu(batch_norm(conv2d(input_ = c0, output_dim = gf_dim * 2, kernel_size = 3, stride = 2, name = 'g_e1_c'), name = 'g_e1_bn'))
        #c2_output: : 1*64*64*256
        c2 = relu(batch_norm(conv2d(input_ = c1, output_dim = gf_dim * 4, kernel_size = 3, stride = 2, name = 'g_e2_c'), name = 'g_e2_bn'))
        
        #residual blocks:
        r1 = residule_block_33(input_ = c2, output_dim = gf_dim*4, atrous = False, name='g_r1')
        r2 = residule_block_33(input_ = r1, output_dim = gf_dim*4, atrous = False, name='g_r2')
        r3 = residule_block_33(input_ = r2, output_dim = gf_dim*4, atrous = False, name='g_r3')
        r4 = residule_block_33(input_ = r3, output_dim = gf_dim*4, atrous = False, name='g_r4')
        r5 = residule_block_33(input_ = r4, output_dim = gf_dim*4, atrous = False, name='g_r5')
        r6 = residule_block_33(input_ = r5, output_dim = gf_dim*4, atrous = False, name='g_r6')
        r7 = residule_block_33(input_ = r6, output_dim = gf_dim*4, atrous = False, name='g_r7')
        r8 = residule_block_33(input_ = r7, output_dim = gf_dim*4, atrous = False, name='g_r8')
        r9 = residule_block_33(input_ = r8, output_dim = gf_dim*4, atrous = False, name='g_r9')
        #resdual block r9 output: 1*64*64*256

        d1 = relu(batch_norm(deconv2d(input_ = r9, output_dim = gf_dim * 2, kernel_size = 3, stride = 2, name = 'g_d1_dc'),name = 'g_d1_bn'))
        d2 = relu(batch_norm(deconv2d(input_ = d1, output_dim = gf_dim, kernel_size = 3, stride = 2, name = 'g_d2_dc'),name = 'g_d2_bn'))
        d3 = conv2d(input_=d2, output_dim  = input_dim, kernel_size = 3, stride = 1, name = 'g_d3_c')
        output = tf.nn.tanh(d3)
        return output

def discriminator(image, gen_label, df_dim=64, reuse=False, name="discriminator"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        dis_concat = tf.concat([image, gen_label], axis=3)
        h0 = lrelu(conv2d(input_ = dis_concat, output_dim = df_dim, kernel_size = 4, stride = 2, name='d_h0_conv'))
        h1 = lrelu(batch_norm(conv2d(input_ = h0, output_dim = df_dim*2, kernel_size = 4, stride = 2, name='d_h1_conv'), 'd_bn1'))
        h2 = lrelu(batch_norm(conv2d(input_ = h1, output_dim = df_dim*4, kernel_size = 4, stride = 2, name='d_h2_conv'), 'd_bn2'))
        h3 = lrelu(batch_norm(conv2d(input_ = h2, output_dim = df_dim*8, kernel_size = 4, stride = 1, name='d_h3_conv'), 'd_bn3'))
        output = conv2d(input_ = h3, output_dim = 1, kernel_size = 4, stride = 1, name='d_h4_conv')
        return output
