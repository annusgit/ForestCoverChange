"""
    Given the path to a single test image (and it's corresponding label if possible), this function generates
    its corresponding segmentation map
"""

from __future__ import print_function
from __future__ import division
import os
import time
import gdal
import random
import shutil
import torch
import numpy as np
np.random.seed(int(time.time()))
random.seed(int(time.time()))
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchnet as tnt
import pickle as pkl
from loss import FocalLoss2d
from model import UNet


def toTensor(**kwargs):
    image, label = kwargs['image'], kwargs['label']
    'will convert image and label from numpy to torch tensor'
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    if kwargs['one_hot']:
        label = label.transpose((2, 0, 1))
        return torch.from_numpy(image).float(), torch.from_numpy(label).float()
    return torch.from_numpy(image).float(), torch.from_numpy(label).long()


def get_inference_loader(image_path, label_path=None, model_input_size=64, num_classes=4, one_hot=False, batch_size=16, num_workers=4):

    # This function is faster because we have already saved our data as subset pickle files
    print('inside dataloading code...')
    class dataset(Dataset):
        def __init__(self, image_path, label_path, stride=model_input_size, bands=[1,2,3], transformation=None):
            super(dataset, self).__init__()
            self.model_input_size = model_input_size
            self.image_path = image_path
            self.all_images = []
            self.total_images = 0
            self.stride = stride
            self.one_hot = one_hot
            self.num_classes = num_classes
            self.transformation = transformation
            self.temp_dir = 'temp_numpy_saves'
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            os.mkdir(self.temp_dir)
            print('LOG: Generating data map now...')
            covermap = gdal.Open(label_path, gdal.GA_ReadOnly)
            channel = covermap.GetRasterBand(1)
            inference_label = np.nan_to_num(channel.ReadAsArray())
            temp_label_path = os.path.join(self.temp_dir, 'temp_label.npy')
            np.save(temp_label_path, inference_label)
            self.temp_test_label = np.load(temp_label_path, mmap_mode='r')
            inference_label, covermap, channel = [None]*3  # release memory
            image_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
            all_raster_bands = [image_ds.GetRasterBand(x) for x in bands]
            cols, rows = image_ds.RasterXSize, image_ds.RasterYSize
            test_image = np.nan_to_num(all_raster_bands[0].ReadAsArray())
            for band in all_raster_bands[1:]:
                test_image = np.dstack((test_image, np.nan_to_num(band.ReadAsArray())))
            temp_image_path = os.path.join(self.temp_dir, 'temp_image.npy')
            np.save(temp_image_path, test_image)
            self.temp_test_image = np.load(temp_image_path, mmap_mode='r')
            test_image, image_ds, all_raster_bands = [None] * 3  # release memory

            row_limit = self.temp_test_label.shape[0] - model_input_size
            col_limit = self.temp_test_label.shape[1] - model_input_size
            for i in range(0, row_limit, self.stride):
                for j in range(0, col_limit, self.stride):
                    self.all_images.append((i, j))
                    self.total_images += 1
            self.shape = [i+self.stride, j+self.stride]
            pass

        def __getitem__(self, k):
            (this_row, this_col) = self.all_images[k]
            this_example_subset = self.temp_test_image[this_row:this_row + self.model_input_size, this_col:this_col + self.model_input_size, :]
            # instead of using the Digital Numbers (DN), use the backscattering coefficient
            HH = this_example_subset[:, :, 0]
            HV = this_example_subset[:, :, 1]
            angle = this_example_subset[:, :, 2]
            HH_gamma_naught = np.nan_to_num(10 * np.log10(HH ** 2 + 1e-7) - 83.0)
            HV_gamma_naught = np.nan_to_num(10 * np.log10(HV ** 2 + 1e-7) - 83.0)
            this_example_subset = np.dstack((HH_gamma_naught, HV_gamma_naught, angle))
            this_label_subset = self.temp_test_label[this_row:this_row + self.model_input_size, this_col:this_col + self.model_input_size, ]
            this_label_subset = (this_label_subset).astype(np.uint8)
            if self.one_hot:
                this_label_subset = np.eye(self.num_classes)[this_label_subset]
            this_example_subset, this_label_subset = toTensor(image=this_example_subset, label=this_label_subset, one_hot=self.one_hot)
            if self.transformation:
                this_example_subset = self.transformation(this_example_subset)
            return {'coordinates': np.asarray([this_row, this_row + self.model_input_size, this_col, this_col + self.model_input_size]),
                    'input': this_example_subset, 'label': this_label_subset}

        def __len__(self):
            return self.total_images

        def get_image_size(self):
            return self.shape

        def clear_mem(self):
            shutil.rmtree(self.temp_dir)
            print('Log: Temporary memory cleared')

    ######################################################################################

    # these are predefined
    # palsar_mean = torch.Tensor([8116.269912828, 3419.031791692, 40.270058337])
    # palsar_std = torch.Tensor([6136.70160067, 2201.432263753, 19.38761076])
    palsar_gamma_naught_mean = [-7.68182243, -14.59668144, 40.44296671]
    palsar_gamma_naught_std = [3.78577892, 4.27134019, 19.73628546]
    transformation = transforms.Compose([transforms.Normalize(mean=palsar_gamma_naught_mean, std=palsar_gamma_naught_std)])

    ######################################################################################

    # create dataset class instances
    # images_per_image means approx. how many images are in each example
    inference_data = dataset(image_path=image_path, label_path=label_path, transformation=transformation) # more images for training
    print('LOG: inference_data ->', len(inference_data))
    inference_loader = DataLoader(dataset=inference_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return inference_loader


@torch.no_grad()
def run_inference(args):
    model = UNet(input_channels=3, num_classes=3)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'), strict=False)
    print('Log: Loaded pretrained {}'.format(args.model_path))
    model.eval()
    # annus/Desktop/palsar/
    test_image_path = '/home/annus/Desktop/palsar/palsar_dataset_full/palsar_dataset/palsar_{}_region_{}.tif'.format(args.year, args.region)
    test_label_path = '/home/annus/Desktop/palsar/palsar_dataset_full/palsar_dataset/fnf_{}_region_{}.tif'.format(args.year, args.region)
    inference_loader = get_inference_loader(image_path=test_image_path, label_path=test_label_path, model_input_size=128, num_classes=4, one_hot=True,
                                            batch_size=args.bs, num_workers=4)
    # we need to fill our new generated test image
    generated_map = np.empty(shape=inference_loader.dataset.get_image_size())
    weights = torch.Tensor([1, 1, 2])
    focal_criterion = FocalLoss2d(weight=weights)
    un_confusion_meter = tnt.meter.ConfusionMeter(2, normalized=False)
    confusion_meter = tnt.meter.ConfusionMeter(2, normalized=True)
    total_correct, total_examples = 0, 0
    net_loss = []
    for idx, data in enumerate(inference_loader):
        coordinates, test_x, label = data['coordinates'].tolist(), data['input'], data['label']
        out_x, softmaxed = model.forward(test_x)
        pred = torch.argmax(softmaxed, dim=1)
        not_one_hot_target = torch.argmax(label, dim=1)
        # convert to binary classes
        # 0-> noise, 1-> forest, 2-> non-forest, 3-> water
        pred[pred == 0] = 2
        pred[pred == 3] = 2
        not_one_hot_target[not_one_hot_target == 0] = 2
        not_one_hot_target[not_one_hot_target == 3] = 2
        # now convert 1, 2 to 0, 1
        pred -= 1
        not_one_hot_target -= 1
        pred_numpy = pred.numpy().transpose(1,2,0)
        for k in range(test_x.shape[0]):
            x, x_, y, y_ = coordinates[k]
            generated_map[x:x_, y:y_] = pred_numpy[:,:,k]
        loss = focal_criterion(softmaxed, not_one_hot_target)  # dice_criterion(softmaxed, label) #
        accurate = (pred == not_one_hot_target).sum().item()
        numerator = float(accurate)
        denominator = float(pred.view(-1).size(0))  # test_x.size(0) * dimension ** 2)
        total_correct += numerator
        total_examples += denominator
        net_loss.append(loss.item())
        un_confusion_meter.add(predicted=pred.view(-1), target=not_one_hot_target.view(-1))
        confusion_meter.add(predicted=pred.view(-1), target=not_one_hot_target.view(-1))
        # if idx % 5 == 0:
        accuracy = float(numerator) * 100 / denominator
        print('{}, {} -> ({}/{}) output size = {}, loss = {}, accuracy = {}/{} = {:.2f}%'.format(args.year,
                                                                                                 args.region,
                                                                                                 idx,
                                                                                                 len(inference_loader),
                                                                                                 out_x.size(),
                                                                                                 loss.item(),
                                                                                                 numerator,
                                                                                                 denominator,
                                                                                                 accuracy))
        #################################
    mean_accuracy = total_correct * 100 / total_examples
    mean_loss = np.asarray(net_loss).mean()
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print('log: test:: total loss = {:.5f}, total accuracy = {:.5f}%'.format(mean_loss, mean_accuracy))
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print('---> Confusion Matrix:')
    print(confusion_meter.value())
    # class_names = ['background/clutter', 'buildings', 'trees', 'cars',
    #                'low_vegetation', 'impervious_surfaces', 'noise']
    with open('normalized.pkl', 'wb') as this:
        pkl.dump(confusion_meter.value(), this, protocol=pkl.HIGHEST_PROTOCOL)
    with open('un_normalized.pkl', 'wb') as this:
        pkl.dump(un_confusion_meter.value(), this, protocol=pkl.HIGHEST_PROTOCOL)

    # save_path = 'generated_maps/generated_{}_{}.npy'.format(args.year, args.region)
    save_path = '/home/annus/Desktop/palsar/generated_maps/using_separate_models/generated_{}_{}.npy'.format(args.year, args.region)
    np.save(save_path, generated_map)
    #########################################################################################3
    inference_loader.dataset.clear_mem()
    pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_path', type=str)
    parser.add_argument('-y', dest='year', type=str)
    parser.add_argument('-r', dest='region', type=str)
    parser.add_argument('-b', dest='bs', type=int)
    args = parser.parse_args()
    run_inference(args)


if __name__ == '__main__':
    main()



















