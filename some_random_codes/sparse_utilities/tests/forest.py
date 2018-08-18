
"""
	Author: Annus Zulfiqar
	Date: 13th May, 2018
	Usage: python forest.py -i "image_name_in_here"
"""

from __future__ import print_function
from __future__ import division

import numpy as np
import argparse
import cv2


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--image', dest='image', help='pass any image you want to find vegetation in...')
	args = parser.parse_args()
	image = args.image
	image = cv2.imread(image)
	new = image.copy()
	blue, green, red = [image[:,:,i] for i in range(3)]
	greenry = 2*green-blue-red
	new[:,:,0] -= greenry
	new[:,:,1] -= greenry
	new[:,:,2] -= greenry
	ret, thresh = cv2.threshold(greenry, 127, 255, cv2.THRESH_OTSU)
	count = np.count_nonzero(thresh)
	print("log: {:.2f}% vegetation found!".format(100*count/(thresh.shape[0]*thresh.shape[1])))
	# cv2.imshow("new", thresh)
	cv2.imshow("image and change", np.hstack((image, new)))
	cv2.waitKey(0)
	pass


if __name__ == '__main__':
	main()





