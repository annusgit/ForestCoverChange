
from __future__ import print_function
from __future__ import division
from os.path import isfile, join
from subprocess import call
import numpy as np
import cv2
import os
from image2gif import writeGif
from PIL import Image


def avi_to_gif(inpath, outpath):
    convert_avi_to_gif = 'ffmpeg -i {} -r 10 -f image2pipe -vcodec ' \
                         'ppm - |convert -delay 5 -loop 0 - {}'.format(inpath, outpath)
    call(convert_avi_to_gif, shell=True)


def convert_frames_to_video(pathIn, pathOut, fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f)) and f.endswith('.png')]

    # for sorting the file names properly
    # files.sort(key=lambda x: int(x[5:-4]))
    files.sort(key=lambda f: int(filter(str.isdigit, f)))
    # sorted(files)
    print('INFO: sort order -> {}'.format(files))
    print('converting now...')
    for i in range(len(files)):
        filename = os.path.join(pathIn, '{}.png'.format(i))
        # reading each files
        img = Image.open(filename)
        if img is None:
            continue
        # height, width, layers = img.shape
        # size = (width, height)
        print(filename)
        # inserting the frames into an image array
        frame_array.append(img)

    writeGif(pathOut, frame_array, duration=0.2)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def main():
    import sys
    pathIn = sys.argv[1]
    pathOut = sys.argv[2]
    fps = int(sys.argv[3])
    convert_frames_to_video(pathIn, pathOut, fps)


if __name__ == "__main__":
    main()






