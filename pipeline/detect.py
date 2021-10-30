# uses the full pipeline functions from patch_classification
# and does end-to-end detection using just the images as the input

# it needs the path to the input tif/png images as input and outputs the segmented grids

from __future__ import print_function
from __future__ import division
import os
import argparse as ap
import matplotlib as mpl
import matplotlib.pyplot as pl
from patch_classification import png_to_pickle as png_to_pickle
from patch_classification.run_model import restore_model, batch_wise_inference
from patch_classification.overlay_prediction_on_image import overlay_with_grid
from patch_classification.change_images_to_video import convert_frames_to_video, avi_to_gif


def do(args):
    inputs_save = os.path.join(args.save_dir, 'images')
    image_save = os.path.join(inputs_save, 'grids')
    down_image_save = os.path.join(inputs_save, 'downsampled')
    label_save = os.path.join(args.save_dir, 'labels')
    if not os.path.exists(inputs_save):
        os.mkdir(inputs_save)
    if not os.path.exists(image_save):
        os.mkdir(image_save)
    if not os.path.exists(down_image_save):
        os.mkdir(down_image_save)
    if not os.path.exists(label_save):
        os.mkdir(label_save)
    model = restore_model(model_name=args.model_name, channels=args.channels,
                          model_path=args.model_path, device=args.device)
    images_list = [x for x in os.listdir(args.images_path) if x.endswith('.png') or x.endswith('.tif')]
    images_list.sort(key=lambda f: int(filter(str.isdigit, f))) # sort the images in the right order
    print('INFO: TESTING IN THE FOLLOWING ORDER: \n', images_list)
    forestation = []
    for count, this_image in enumerate(images_list, 1):
        print('INFO: ON IMAGE >', this_image)
        full_path = os.path.join(args.images_path, this_image)
        # print(full_path)
        # saves the image as pkl temporarily
        png_to_pickle(image_file=full_path, pkl_file='tmp.pkl', bands=args.bands)
        # this function returns shape of our used image and raw forest percentage
        (H, W, C), forest_percentage = batch_wise_inference(model=model, data_type=args.data_type,
                                                            image_path='tmp.pkl',
                                                            batch_size=20, device=args.device,
                                                            number='tmp', count=count,
                                                            total=len(images_list))
        # and this returns a refined value of forestation
        filtered_forest_percentage = overlay_with_grid(image_path='test_image_tmp.npy',
                                                       pred_path='image_pred_tmp.npy',
                                                       data_type=args.data_type,
                                                       image_save_path=os.path.join(image_save,
                                                                                    '{}.png'.format(count)),
                                                       downsampled_image_save_path=os.path.join(down_image_save,
                                                                                    '{}.png'.format(count)),
                                                       label_save_path=os.path.join(label_save,
                                                                                    '{}.png'.format(count)),
                                                       shape=(H,W,C if C == 3 else 3),
                                                       type=args.data_type)
        print('\nINFO: {}% Forestation Found.'.format(filtered_forest_percentage))
        forestation.append(filtered_forest_percentage) # forest_percentage
    print(image_save, os.path.join(image_save, 'out.avi'))
    convert_frames_to_video(pathIn=image_save, pathOut=os.path.join(image_save, 'out.avi'), fps=2)
    convert_frames_to_video(pathIn=down_image_save, pathOut=os.path.join(down_image_save, 'out.avi'), fps=2)
    convert_frames_to_video(pathIn=label_save, pathOut=os.path.join(label_save, 'out.avi'), fps=2)
    # call(convert_avi_to_gif, shell=True)
    # call(convert_avi_to_gif, shell=True)
    # call(convert_avi_to_gif, shell=True)
    avi_to_gif(inpath=os.path.join(image_save, 'out.avi'), outpath=os.path.join(image_save, 'out.gif'))
    avi_to_gif(inpath=os.path.join(down_image_save, 'out.avi'), outpath=os.path.join(down_image_save, 'out.gif'))
    avi_to_gif(inpath=os.path.join(image_save, 'out.avi'), outpath=os.path.join(label_save, 'out.gif'))
    # make a graph of the forestation change
    print(forestation)
    figure = pl.figure()
    mpl.rcParams.update(mpl.rcParamsDefault)
    pl.plot(forestation, 'go-', label='forest_percentage')
    pl.title('forestation trend (2015-2018)')
    pl.xlabel('years')
    pl.ylabel('forest percentage (%)')
    pl.legend(loc='upper right')
    pl.axis('off')
    pl.savefig(os.path.join(args.save_dir, 'forestation_plot.png'))
    # pl.show()
    # cleanup
    os.remove('test_image_tmp.npy')
    os.remove('image_pred_tmp.npy')
    os.remove('tmp.pkl')
    pass


def main():
    args = ap.ArgumentParser()
    args.add_argument('--images', dest='images_path')
    args.add_argument('--data', dest='data_type')
    args.add_argument('--model_type', dest='model_name')
    args.add_argument('--channels', type=int, dest='channels')
    args.add_argument('--trained_model', dest='model_path')
    args.add_argument('--bands', nargs='+', type=int, dest='bands')
    args.add_argument('--save_dir', dest='save_dir')
    args.add_argument('--device', dest='device')
    arguments = args.parse_args()
    if not os.path.exists(arguments.save_dir):
        os.mkdir(arguments.save_dir)
    do(args=arguments)


if __name__ == '__main__':
    main()












