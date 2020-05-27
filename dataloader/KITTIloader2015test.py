import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

  left_fold  = 'image_2/'
  right_fold = 'image_3/'
  disp_L = 'disp_occ_0/'
  disp_R = 'disp_occ_1/'

  image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]

  left_val  = [filepath+left_fold+img for img in image[:40]]
  right_val = [filepath+right_fold+img for img in image[:40]]
  disp_val_L = [filepath+disp_L+img for img in image[:40]]
  #disp_val_R = [filepath+disp_R+img for img in image]

  return left_val, right_val, disp_val_L
