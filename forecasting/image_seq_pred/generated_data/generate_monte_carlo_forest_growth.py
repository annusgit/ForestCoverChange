

from __future__ import print_function
from __future__ import division
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pl
# random.seed(int(time.time()))


def get_neighbour(img, pixel, pix_val=0):
    x_max, y_max = img.shape
    i, j = pixel
    possible_neighbours = [(i - 1, j - 1), (i, j - 1), (i + 1, j - 1),
                           (i - 1, j)                , (i + 1, j),
                           (i - 1, j + 1), (i, j + 1), (i + 1, j + 1)]
    # those that are within legitimate boundaries of the image
    legitimate_neighbours = [(x,y) for (x,y) in possible_neighbours if 0 <= x < x_max and 0 <= y < y_max]
    # growing_neighbours = [(x,y) for (x,y) in legitimate_neighbours if img[x,y] == pix_val]
    # if len(growing_neighbours) != 0:
    #     return growing_neighbours[np.random.choice(len(growing_neighbours))]
    return legitimate_neighbours[np.random.choice(len(legitimate_neighbours))]


def generate():
    pl.figure(1)
    rows, cols = (256, 256)
    c = mpl.colors.ListedColormap(['white', 'green'])
    n = mpl.colors.Normalize(vmin=0, vmax=1)
    # random = np.random.rand()
    # img = np.random.choice([0, 1], size=(rows, cols), p=[random, 1 - random])
    # img1 = np.zeros(shape=(rows, cols//2))
    # img2 = np.ones(shape=(rows, cols//2))
    # img = img + np.hstack((img1,img2))
    # print(img.shape)
    img = np.zeros(shape=(rows, cols))
    # make random shape inside it
    # shapes of different sizes
    circles = []
    for i in range(50):
        circles.append((1, 1, np.random.randint(rows), np.random.randint(cols), np.random.randint(20)))
    for y in range(rows):
        for x in range(cols):
            # check if it lies within any one of our circles
            for (a,b,h,k,r) in circles:
                if (x/a-h)**2 + (y/b-k)**2 < r**2:
                    # print('making it 1')
                    img[y,x] = 1
    # start sequence generation
    for frame in range(0, 300): # outer loop generates new frames
        start_t = time.time()
        # inner loops generate growth pattern based on
        # https://www.gamedev.net/blogs/entry/2249737-another-cellular-automaton-video/
        for i in range(rows):
            for j in range(cols):
                if img[i,j] == 1:
                    grow_this_one = get_neighbour(img=img, pixel=(i,j), pix_val=0)
                    img[grow_this_one] = 1
        t_elapsed = time.time()-start_t
        pl.clf()
        pl.imshow(img, cmap=c, norm=n)
        pl.title('Number {} (time elapsed for generation = {:.2f} seconds)'.format(frame, t_elapsed))
        pl.pause(0.01)


if __name__ == '__main__':
    generate()


