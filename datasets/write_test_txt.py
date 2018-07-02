import numpy as np
import glob
import os
import random

def main():
    test_input_dir = "./test/X/images/"
    test_txt_path = "./x_testdata.txt"
    testf = file(test_txt_path, 'a+')
    test_input_images = glob.glob(os.path.join(test_input_dir, "*.jpg")) #return a list of x
    for idx in range(len(test_input_images)):
        test_name, _ = os.path.splitext(os.path.basename(test_input_images[idx]))
        test_domain_name, test_idx = test_name.split('Sight')
        test_image_name = test_name + '.jpg'
        test_label_name = test_domain_name+'Label'+test_idx+'.png'
        test_content = test_image_name+" "+test_label_name+"\n"
        testf.write(test_content)
    testf.close()

if __name__ == '__main__':
    main()
