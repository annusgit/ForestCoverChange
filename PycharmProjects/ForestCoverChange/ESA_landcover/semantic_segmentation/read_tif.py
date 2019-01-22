
from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as pl
from scipy import misc
import numpy as np


def main():
	this = misc.imread('test.tif')
	print(np.max(this))
	for i in range(100):
		rand = np.random.randint(0,this.shape[0]-64)
		new = this[rand:rand+64,rand:rand+64,:4]
		pl.imshow(new)
		pl.show()
	pass


if __name__ == '__main__':
	main()

