

# uses the full pipeline functions from patch_classification
# and does end-to-end detection using just the images

# it needs the path to the input tif/png images as input and outputs the segmented grids

from __future__ import print_function
from __future__ import division
import os
import argparse as ap
from patch_classification import png_to_pickle as png_to_pickle
from patch_classification.run_model import restore_model, batch_wise_inference
from patch_classification.overlay_prediction_on_image import overlay_with_grid
from patch_classification.change_images_to_video import convert_frames_to_video


def do(**kwargs):
    images_path = kwargs['images_path']
    model = kwargs['model']
    device = kwargs['device']
    save = kwargs['save']
    model = restore_model(model_path=model, device=device)
    images_list = [x for x in os.listdir(images_path) if x.endswith('.png') or x.endswith('.tif')]
    for count, this_image in enumerate(images_list, 1):
        full_path = os.path.join(images_path, this_image)
        # print(full_path)
        # saves the image as pkl temporarily
        png_to_pickle(image_file=full_path, pkl_file='tmp.pkl')
        (H, W, C) = batch_wise_inference(model=model, image_path='tmp.pkl', device=device, number='tmp')
        overlay_with_grid(image_path='test_image_tmp.npy',
                          pred_path='image_pred_tmp.npy',
                          save_path=os.path.join(save, '{}.png'.format(count)),
                          shape=(H,W,C))
    convert_frames_to_video(pathIn=save, pathOut=os.path.join(save, 'out.avi'), fps=2)
    os.remove('test_image_tmp.npy')
    os.remove('image_pred_tmp.npy')
    os.remove('tmp.pkl')
    pass


def main():
    args = ap.ArgumentParser()
    args.add_argument('--images', dest='images_path')
    args.add_argument('--model', dest='model_path')
    args.add_argument('--save_dir', dest='save_dir')
    args.add_argument('--device', dest='device')
    arguments = args.parse_args()
    images_path = arguments.images_path
    model = arguments.model_path
    device = arguments.device
    save = arguments.save_dir
    if not os.path.exists(save):
        os.mkdir(save)
    do(images_path=images_path, model=model, device=device, save=save)


if __name__ == '__main__':
    main()












