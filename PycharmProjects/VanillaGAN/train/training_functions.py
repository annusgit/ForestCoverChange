

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as opt
from torch.nn.utils import clip_grad_norm_
from models import judge, intelligence
import matplotlib.pyplot as pl
import numpy as np
from data import get_dataloaders
import os


def train(**kwargs):
    """
        This will be a two step process, first train disc only, then train gen only
    :param kwargs:
    :return:
    """
    batch_size = kwargs['batch_size']
    gen = kwargs['gen']
    disc = kwargs['discriminator']
    device = kwargs['device']
    gen.to(device)
    disc.to(device)
    lr = kwargs['lr']
    iters = int(kwargs['iterations'])
    save_after = int(kwargs['save_after'])
    models_dir = kwargs['save_dir']
    try:
        os.mkdir(models_dir)
    except:
        pass
    half = int(batch_size/2)
    data_loader, _ = get_dataloaders(batch_size=half)
    disc_optimizer = opt.Adam(params=disc.parameters(), lr=lr)
    gen_optimizer = opt.Adam(params=gen.parameters(), lr=lr)
    criterion = nn.BCELoss()
    real, fake = 0, 1 # labels
    ########################### train discriminator ############################
    print('INFO: Training the discriminator (Generator frozen)')
    for k in range(iters):
        batch_loss, batch_correct = 0, 0
        for j, (real_images, _) in enumerate(data_loader):
            if real_images.shape[0] != half:
                continue # skip the last part
            # train on real and fake images alike
            real_images = real_images.to(device)
            noise = torch.Tensor(half, 1).to(device)
            # freeze the generator for now
            with torch.no_grad():
                gen.eval()
                gen_images = gen(noise).to(device)
            disc.zero_grad() # it's important to not to accumulate any gradients
            random_permutes = torch.randperm(batch_size)
            batch_images = torch.cat((real_images, gen_images), dim=0)
            batch_labels = torch.cat((torch.Tensor(half).fill_(real), torch.Tensor(half).fill_(fake)), dim=0)
            batch_images = batch_images[random_permutes].to(device)
            batch_labels = batch_labels[random_permutes].to(device)
            batch_out = disc(batch_images)
            batch_out = torch.argmax(batch_out, dim=1).float()
            # print(batch_out.dtype, batch_labels.dtype)
            loss = criterion(batch_out, batch_labels)
            batch_correct += batch_out.eq(batch_labels.view_as(batch_out)).double().sum().item()/batch_size
            batch_loss += loss.item()
            # print(gen_images.shape)
            ############## order of steps is important
            loss.backward()
            clip_grad_norm_(disc.parameters(), max_norm=0.5)
            disc_optimizer.step()
            ##############
            if j % 10 == 0:
                print('log: epoch {}: '.format(k)+'({})/({})'.format(j, len(data_loader)))
        print('\n({})/({}) loss = {}, accuracy = {}'.format(k+1, iters,
                                                            batch_loss/len(data_loader),
                                                            batch_correct*100/len(data_loader)))
        torch.save(disc.state_dict(), os.path.join(models_dir, 'model-{}.pt'.format(k+1)))

    ########################### train generator ################################
    print('INFO: Training the generator (Discriminator frozen)')
    # for k in range(iters):
    #     batch_loss, batch_correct = 0, 0
    #     for j, (real_images, _) in enumerate(data_loader):
    #         # train on real and fake images alike
    #         noise = torch.Tensor(half, 1)
    #         gen_images = gen(noise)
    #         # freeze the discriminator now
    #         with torch.no_grad():
    #             disc.eval()
    #         gen.zero_grad() # it's important to not to accumulate any gradients
    #         random_permutes = torch.randperm(batch_size)
    #         batch_images = torch.cat((real_images, gen_images), dim=0)
    #         batch_labels = torch.cat((torch.Tensor(half).fill_(real), torch.Tensor(half).fill_(fake)), dim=0)
    #         batch_images = batch_images[random_permutes]
    #         batch_labels = batch_labels[random_permutes]
    #         batch_out = disc(batch_images)
    #         batch_out = torch.argmax(batch_out, dim=1).float()
    #         # print(batch_out.dtype, batch_labels.dtype)
    #         loss = criterion(batch_out, batch_labels)
    #         batch_correct += batch_out.eq(batch_labels.view_as(batch_out))
    #         batch_loss += loss.item()
    #         # print(gen_images.shape)
    #         print('\r'+'log: epoch {}: '.format(k)+'|'*j+'({})/({})'.
    #               format(j, len(data_loader)), end='')
    #     print('\n({})/({}) loss = {}, accuracy = {}'.format(k+1, iters, batch_loss/iters, batch_correct/iters))
    #     torch.save(disc.state_dict(), os.path.join(models_dir, 'model-{}.pt'.format(k+1)))
    #
    #     # if k % save_after == 0 and k > 0:
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
    batch_size = 32
    train(batch_size=batch_size, gen=gen, discriminator=disc,
          lr=1e-4, iterations=100, save_after=10, save_dir='saved_models',
          device='cpu')
    pass


if __name__ == '__main__':
    train_tst()



