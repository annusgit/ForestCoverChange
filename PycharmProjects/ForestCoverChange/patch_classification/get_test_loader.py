


from __future__ import print_function
from __future__ import division
from subprocess import call
import pickle as p


def save_test_images():
    save_dest = '/home/annus/Desktop/test_images/'
    try:
        call('mkdir {}'.format(save_dest), shell=True)
    except:
        pass
    with open('test_loader.pkl', 'rb') as test:
        test_images = p.load(test)
    # print(test_images)
    for key in test_images.keys():
        image_path, label = test_images[key]
        call('cp {} {}'.format(image_path, save_dest), shell=True)
        print(image_path)
    pass


if __name__ == '__main__':
    save_test_images()