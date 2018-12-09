

"""
    Test a pretrained patch network on a real sentinel image
"""

from __future__ import print_function
from __future__ import division
import torch
import sys
import cv2
# import gdal
import imageio
import numpy as np
from model import *
from scipy import misc
import matplotlib.pyplot as pl
from dataset import get_inference_loader


@torch.no_grad()
def test_model_on_real_sentinel_image():
    net = ResNet(in_channels=3)
    test_model = sys.argv[1] # '/home/annus/Desktop/test.tif'
    image_path = sys.argv[2] # image_pred = np.zeros_like(image_read[:,:,0])
    device = sys.argv[3]
    net.load_state_dict(torch.load(test_model))
    # torch.save(net.state_dict(), '/home/annus/Desktop/trained_resnet_cpu.pth')
    net.to(device=device)
    net.eval()

    # bands = [3,2,1]
    # this = gdal.Open(image_path)
    # image_read = this.GetRasterBand(bands[0]).ReadAsArray()
    # for i in bands[1:]:
    #     image_read = np.dstack((image_read,
    #                             this.GetRasterBand(i).ReadAsArray())).astype(np.int16)

    patch = 64 # model input is of fixed size, 64, so...
    test_size = 512
    # ds_r = gdal.Open(red_path)
    # ds_g = gdal.Open(green_path)
    # ds_b = gdal.Open(blue_path)
    # r = np.array(ds_r.GetRasterBand(1).ReadAsArray())
    # g = np.array(ds_g.GetRasterBand(1).ReadAsArray())
    # b = np.array(ds_b.GetRasterBand(1).ReadAsArray())
    image_read = cv2.imread(image_path, -1)[:,:,:3].astype(np.float32)/4096.0
    image_read = np.dstack((image_read[:,:,2], image_read[:,:,1], image_read[:,:,0]))
    print(image_read.shape, image_read.max(), image_read.min())
    # pl.imshow((255*image_read).astype(np.uint8))
    # pl.show()
    # image_read = np.asarray(image_read.astype(np.uint16))
    # print(image_read.dtype, np.max(image_read), image_read.min())
    # image_read = image_read.astype(np.float) #/255 no need since matplot already does that...
    # image_read = image_read[:,:,:3]
    # print(image_read.max())
    # pl.imshow(image_read)
    # pl.show()
    image_pred = np.zeros_like(image_read[:,:,0])

    def toTensor(image):
        "converts a single input image to tensor"
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image).float().unsqueeze(0)

    # this is a very bad approach...
    for i in range(image_read.shape[0]//patch):
        for j in range(image_read.shape[1]//patch):
            image = image_read[patch*i:patch*i+patch, j*patch:j*patch+patch, :]
            # print(np.unique(image))
            print('log: w = {}-{}, h = {}-{}'.format(patch*i, patch*i+patch, j*patch, j*patch+patch), end=' ')
            # test_x = torch.from_numpy(image.transpose(2,0,1)).unsqueeze(0).float().cuda()
            test_x = toTensor(image).to(device=device)
            # print(test_x.shape)
            out_x, pred = net(test_x)
            pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)
            image_pred[patch*i:patch*i+patch, j*patch:j*patch+patch] = pred
            print(pred)

    cv2.imwrite('pred_sentinel_muzaffarabad.png', image_pred)
    cv2.imwrite('image_test_muzaffarabad.png', (255*image_read).astype(np.uint8))


def restore_model(model_name, channels, model_path, device):
    # net = VGG_N(in_channels=5)
    net = eval(model_name)(in_channels=channels)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device=device)
    net.eval()
    return net


def batch_wise_inference(model, data_type, image_path, batch_size, device, number, count, total):
    inference_loader, (H,W,C) = get_inference_loader(image_path=image_path, batch_size=batch_size)
    # test_image = np.zeros(shape=(H,W,C)) # for saving the image
    # image_pred = np.zeros(shape=(H,W))
    # np.save('test_image.npy', test_image)
    # np.save('image_pred.npy', image_pred)
    # del test_image, image_pred
    image_C = C
    if C > 3:
        image_C = 3
    test_image = np.memmap('test_image_{}.npy'.format(number), dtype=np.uint16, mode='w+', shape=(H,W,image_C))
    image_pred = np.memmap('image_pred_{}.npy'.format(number), dtype=np.uint8, mode='w+', shape=(H,W))

    # this is a much better approach, batch wise inference...
    # also calculate the forest % in each image
    forest_count = 0
    forest_label = 0 if data_type == 'pakistan' else 1
    for idx, data in enumerate(inference_loader, 1):
        log_str = 'log: image ({}/{})'.format(count, total)+'-'*int(idx/len(inference_loader)*50)+\
                  '> batch ({}/{})'.format(idx, len(inference_loader))
        sys.stdout.write('\r'+log_str)
        sys.stdout.flush()
        test_x, indices = data['input'], data['indices']
        indices = indices.numpy() # bring the indices to the cpu
        test_x.to(device=device)
        out_x, pred = model(test_x)
        pred = pred.numpy().astype(np.int)
        forest_count += (pred == forest_label).sum().item()   # manually insert the label of forests
        test_x = (test_x.numpy()).transpose(0, 2, 3, 1)
        # print(test_x.max())
        # test_x = (test_x*255).astype(np.uint8)
        # convert test image tensor to rgb
        # test_x range [-1,1]
        test_x = (test_x + 1)  # range [0,2]
        test_x /= 2  # range [0,1]
        test_x *= 255  # range [0, 255] but float
        test_x = test_x.astype(np.uint8)
        x1, x2, y1, y2 = indices[:,0], indices[:,1], indices[:,2], indices[:,3]
        for k in range(len(x1)):
            rgb = test_x[k,:,:,:]
            rgb = rgb[:,:,[2,1,0]]
            test_image[x1[k]:x2[k],y1[k]:y2[k],:] = rgb[:,:,:3] #
            # print(test_image[x1[k]:x2[k],y1[k]:y2[k],:])
            image_pred[x1[k]:x2[k],y1[k]:y2[k]] = pred[k]
            # print(pred[k])
    forest_percent = 100*forest_count/(len(inference_loader)*batch_size)
    # we will need the shape later on as well as the forest percentage
    return (H, W, C), forest_percent


if __name__ == '__main__':
    print(batch_wise_inference(model=restore_model(model_path='/home/annus/PycharmProjects/'
                                                              'ForestCoverChange_inputs_and_numerical_results/'
                                                              'patch_wise/trained_resnet_cpu.pth', device='cpu'),
                               type='germany', image_path='test.pkl', batch_size=20,
                               device='cpu', number='tmp',
                               count=1, total=1))





