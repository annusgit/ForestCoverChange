

from __future__ import print_function, division
import time
import numpy as np
import matplotlib.pyplot as pl


def get_neighbour(img, pixel, shape):
    x_max, y_max = shape
    i, j = pixel
    possible_neighbours = [(i - 1, j - 1), (i, j - 1), (i + 1, j - 1),
                           (i - 1, j)                , (i + 1, j),
                           (i - 1, j + 1), (i, j + 1), (i + 1, j + 1)]
    # those that are within legitimate boundaries of the image
    legitimate_neighbours = [(x,y) for (x,y) in possible_neighbours
                             if x >= 0 and x < x_max and y >= 0 and y < y_max]
    growing_neighbours = [(x,y) for (x,y) in legitimate_neighbours if img[x,y] == 0]
    print(len(growing_neighbours))
    if len == 0:
        return np.random.choice(legitimate_neighbours)
    return np.random.choice(growing_neighbours)


def generate():
    pl.figure(1)
    rows, cols = (256, 256)
    random = np.random.rand()
    # img = np.random.choice([0, 1], size=(rows, cols), p=[random, 1 - random])
    img1 = np.zeros(shape=(rows, cols//2))
    img2 = np.ones(shape=(rows, cols//2))
    img = np.hstack((img1,img2))
    print(img.shape)
    for frame in range(0, 300): # outer loop generates new frames
        start_t = time.time()
        # inner loops generate growth pattern based on
        # https://www.gamedev.net/blogs/entry/2249737-another-cellular-automaton-video/
        for i in range(rows):
            for j in range(cols):
                if img[i,j] == 1:
                    grow_this_one = get_neighbour(img=img, pixel=(i,j), shape=(rows, cols))
                    img[grow_this_one] = 1
        t_elapsed = time.time()-start_t
        pl.clf()
        pl.imshow(img)
        pl.title('Number {} (time elapsed for generation = {:.2f} seconds)'.format(frame, t_elapsed))
        pl.pause(0.05)


if __name__ == '__main__':
    generate()


