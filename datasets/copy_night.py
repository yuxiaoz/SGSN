import numpy as np
import glob
import os
import random
import cv2

def main():
    night_image_dir = "./train/X/images/"
    night_label_dir = "./train/X/labels/"
    night_images = glob.glob(os.path.join(night_image_dir, "Night*.jpg"))
    night_labels = glob.glob(os.path.join(night_label_dir, "Night*.png"))
    for idx in range(len(night_images)):
        night_image_name, _ = os.path.splitext(os.path.basename(night_images[idx]))
        night_label_name, _ = os.path.splitext(os.path.basename(night_labels[idx]))
        night_image_name_head, night_image_name_tail = night_image_name.split('Sight')
        night_label_name_head, night_label_name_tail = night_label_name.split('Label')
        night_image = cv2.imread(night_images[idx], -1)
        night_label = cv2.imread(night_labels[idx], -1)
        copy_night_image_path = night_image_dir + night_image_name_head + 'SightTwo' + night_image_name_tail + '.jpg'
        copy_night_label_path = night_label_dir + night_label_name_head + 'LabelTwo' + night_label_name_tail + '.png'
        cv2.imwrite(copy_night_image_path, night_image)
        cv2.imwrite(copy_night_label_path, night_label)
        print(idx)

if __name__ == '__main__':
    main()
