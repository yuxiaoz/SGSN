from PIL import Image
import numpy as np
import tensorflow as tf

def label_proc(label):
    return ((-1. * (label - 1.)) * 255.).astype(np.float32)

def label_inv_proc(label):
    return ((-1. * (label - 1.)) * 255.)[:,:,:,::-1].astype(np.float32)

def cv_inv_proc(img):
    img_rgb = (img + 1.) * 127.5
    return img_rgb[:,:,:,::-1].astype(np.float32) #rgb
   
