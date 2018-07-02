import numpy as np
import glob
import os
import random

def main():
    x_input_dir = "./train/X/images/"
    y_input_dir = "./train/Y/images/"
    x_txt_path = "./x_traindata.txt"
    y_txt_path = "./y_traindata.txt"
    xf = file(x_txt_path, 'a+')
    yf = file(y_txt_path, 'a+')
    x_input_images = glob.glob(os.path.join(x_input_dir, "*.jpg")) #return a list of x
    y_input_images_t = glob.glob(os.path.join(y_input_dir, "*.jpg")) #return a list of y
    mul_num = int(len(x_input_images)/len(y_input_images_t))
    y_append_num = len(x_input_images) - len(y_input_images_t)*mul_num
    append_list = [random.randint(0,len(y_input_images_t)-1) for i in range(y_append_num)]
    y_append_images = []
    for a in append_list:
        y_append_images.append(y_input_images_t[a])
    y_input_images = y_input_images_t * mul_num + y_append_images
    random.shuffle(x_input_images)
    random.shuffle(y_input_images)
    for idx in range(len(x_input_images)):
        x_name, _ = os.path.splitext(os.path.basename(x_input_images[idx]))
        y_name, _ = os.path.splitext(os.path.basename(y_input_images[idx]))
        x_domain_name, x_idx = x_name.split('Sight')
        x_image_name = x_name + '.jpg'
        x_label_name = x_domain_name+'Label'+x_idx+'.png'
        y_domain_name, y_idx = y_name.split('Sight')
        y_image_name = y_name + '.jpg'
        y_label_name = y_domain_name+'Label'+y_idx+'.png'
        x_content = x_image_name+" "+x_label_name+"\n"
        y_content = y_image_name+" "+y_label_name+"\n"
        xf.write(x_content)
        yf.write(y_content)
    xf.close()
    yf.close()

if __name__ == '__main__':
    main()
