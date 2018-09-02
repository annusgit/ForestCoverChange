

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as opt
from models import judge, intelligence
import matplotlib.pyplot as pl
import numpy as np
from data import get_dataloaders


def train(**kwargs):
    """
        This will be a two step process, first train disc only, then train gen only
    :param kwargs:
    :return:
    """
    batch_size = kwargs['batch_size']
    gen = kwargs['gen']
    disc = kwargs['discriminator']
    lr = kwargs['lr']
    iters = int(kwargs['iterations'])
    save_after = int(kwargs['save_after'])
    half = int(batch_size/2)
    data_loader, _ = get_dataloaders(batch_size=half)
    disc_optimizer = opt.Adam(params=disc.parameters(), lr=lr)
    gen_optimizer = opt.Adam(params=gen.parameters(), lr=lr)
    criterion = nn.BCELoss()
    real, fake = 1, 0 # labels
    ########################### train discriminator ############################
    for k in range(iters):
        batch_loss, batch_correct = 0, 0
        for real_images, _ in data_loader:
            # train on real and fake images alike
            noise = torch.Tensor(half, 1)
            # freeze the generator for now
            with torch.no_grad():
                gen.eval()
                gen_images = gen(noise)
            disc.zero_grad() # it's important to not to accumulate any gradients
            random_permutes = torch.randperm(batch_size)
            batch_images = torch.cat((real_images, gen_images), dim=0)
            batch_labels = torch.cat((torch.Tensor(half).fill_(1), torch.Tensor(half).fill_(0)), dim=0).long()
            batch_images = batch_images[random_permutes]
            batch_labels = batch_labels[random_permutes]
            batch_out = disc(batch_images)
            loss = criterion(batch_out, batch_labels)
            batch_correct += batch_out.eq(batch_labels.view_as(batch_out))
            batch_loss += loss.item()
            # print(gen_images.shape)


    ########################### train generator ################################

        # if k % save_after == 0 and k > 0:
        #     images, results = gen_images.detach().numpy(), torch.argmax(discriminated_probs, dim=1).detach().numpy()
        #     images = images.squeeze(1).transpose(0, 1, 2)
        #     # print(np.unique(images))\
        #     for j in range(16):
        #         pl.subplot(4, 4, j + 1)
        #         # print(images.shape)
        #         pl.imshow(images[j, :, :])
        #         pl.title(results[j])
        #         pl.axis('off')
        #     pl.show()
    pass


def train_tst():
    gen, disc = intelligence(), judge()
    batch_size = 16
    train(batch_size=batch_size, gen=gen, discriminator=disc, lr=1e-4, iterations=100, save_after=10)
    pass


if __name__ == '__main__':
    train_tst()



