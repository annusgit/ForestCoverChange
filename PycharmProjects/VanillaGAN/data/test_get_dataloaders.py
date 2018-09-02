

from unittest import TestCase
from data import get_dataloaders
import matplotlib.pyplot as pl

class TestGet_dataloaders(TestCase):
    def test_get_dataloaders(self):
        train, test = get_dataloaders(batch_size=16)
        for k, data in enumerate(train):
            images, labels = data
            print(images.shape, labels.shape)
            images, labels = images.numpy(), labels.numpy()
            images = images.squeeze(1).transpose(0,1,2)
            for j in range(16):
                pl.subplot(4,4,j+1)
                print(images.shape)
                pl.imshow(images[j,:,:])
                pl.title(labels[j])
                pl.axis('off')
            pl.show()



