from torchvision import datasets, transforms
import PIL
import torch
from torch.utils.data import Dataset, DataLoader
import os
from skimage import io, transform
import skimage
import numpy as np
import pandas as pd
import kornia as K
import random


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        T = transforms.ToTensor()
        image = T(image)
        if not landmarks is 0:
          landmarks = torch.tensor(landmarks,dtype=torch.float32)
        return (image,landmarks)


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample

        h, w = image.size[:2]
        new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        m = transforms.Resize((new_h, new_w), PIL.Image.BICUBIC)
        img = m(image)

        return (img,landmarks)

        

class ToGray(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample
        G = transforms.Grayscale(num_output_channels=1)
        if image.size[0]>1:
          image = G(image)
        return (image,landmarks)



class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image."""

    def __call__(self, sample):
          image, landmarks = sample
          col = transforms.ColorJitter(brightness=(0.7,1.3))
          image = col(image)
          return (image,landmarks)   



class GaussianBlur(object):
    """Blurs image with randomly chosen Gaussian blur"""

    def __call__(self, sample):
          image, landmarks = sample
          kernel = random.randint(1,5)
          image = image.filter(PIL.ImageFilter.GaussianBlur(radius=kernel))
          return (image,landmarks)  



params = {'deg':5, 'transx':0.2, 'transy':0.2, 'minscale':1.2, 'maxscale':1.5, 'shear':(-15,15)}
def get_twin(im,params=params):
  image_tens = im.type(torch.float32)
  h=im.size()[2]
  w=im.size()[3]
  #warp parameters
  degrees = (-params['deg'],params['deg'])
  translate = (params['transx'],params['transy'])
  scale = (params['minscale'],params['maxscale'])
  shear = params['shear']

  #get warped images and homographies 
  aug = K.augmentation.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear ,  return_transform=True)
  image_tens_warp, H = aug(image_tens)
  return image_tens_warp, H
