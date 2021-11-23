
from tensorflow.keras.layers import BatchNormalization
from keras.callbacks import TensorBoard


from keras.utils.vis_utils import plot_model
from skimage.color import rgb2lab, lab2rgb, rgb2gray,gray2rgb
from skimage.io import imsave, imread, imshow

from skimage.util import random_noise
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as sk_ssim, peak_signal_noise_ratio as sk_psnr
from tqdm import tqdm
from pathlib import Path
from functools import lru_cache
from utils.utils import load_im
from utils.utils import np_resc
from utils.models import create_model_alpha
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


class colorization():
  def __init__(self, *, path = None, img_size = 256, initial_frame = 0, final_frame = None):
    self.path = Path(path)
    self.images = load_im(self.path)
  def __call__():
    self.model = create_model_alpha()
    for i in range():


    pass;
