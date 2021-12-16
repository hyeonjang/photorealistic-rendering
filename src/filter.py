import argparse
import torch

import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

import numpy as np
import cv2

from util import render_util as ru
from config import CONFIGS

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='test_radiative_backprop.py')
    parser.add_argument('input_name', type=str, help='output directories')

    args = parser.parse_args()
    loss_function = torch.nn.MSELoss()


    image = ru.image_to_torch(args.input_name)

    gaussian = cv2.getGaussianKernel(1, 1)
    kernel = np.outer(gaussian, gaussian.transpose())
    filtered = cv2.filter2D(image.numpy(), -1, 50*kernel)

    ru.write_image("tmp.png", image)
