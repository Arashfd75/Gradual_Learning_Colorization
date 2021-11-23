from pathlib import Path
import os
import numpy as np
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from math import log10, sqrt


def load_im(path, shift, size,number_to_load=None, filename_return=False,split_phrase = 'frame'):
  output = []
  number_of_images = 0
  list_of_files = os.listdir(path)
  list_of_files = sorted(list_of_files, key=lambda f: int(f.split(f'{split_phrase}')[1].split('.')[0]))
  for filename in list_of_files:
      if number_of_images < shift:
        number_of_images += 1
        continue
      if number_to_load==None or number_of_images < number_to_load + shift:
        number_of_images += 1
        # print(filename)
        # /content/drive/MyDrive/colabData_1/
        name = Path(path) / filename
        #str(filename)
        original = resize(img_to_array(load_img(name)),(size,size))
        output.append(original)
        if filename_return:
          return filename
      else:
        break;

  return output


def np_resc(X):
  X = np.array(X, dtype=float)

  # Set up train and test data
  split = int(1*len(X))
  # split = int(0.05*len(X))
  Xtrain = X[:split]
  Xtrain = 1.0/255*Xtrain
  Xtrain.shape
  return Xtrain


def PSNR(original, compressed):
    # original = original * 255
    # compressed = compressed * 255
    # original = original.astype(np.uint8)
    # compressed = compressed.astype(np.uint8)
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr