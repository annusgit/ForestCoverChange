
from __future__ import print_function
from __future__ import division
from os.path import isfile, join
import numpy as np
import cv2
import os


def convert_frames_to_video(pathIn, pathOut, fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    # for sorting the file names properly
    # files.sort(key=lambda x: int(x[5:-4]))
    sorted(files)

    for i in range(len(files)):
        filename = pathIn + files[i]
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        print(filename)
        # inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def main():
    pathIn = '/home/annus/PycharmProjects/ForestCoverChange/results/color_coded_segmentation/'
    pathOut = 'color_coded_seg.avi'
    fps = 3.0
    convert_frames_to_video(pathIn, pathOut, fps)


if __name__ == "__main__":
    main()






