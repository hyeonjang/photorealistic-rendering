import argparse
import torch

import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

from util import render_util as ru
from config import CONFIGS

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='test_radiative_backprop.py')
    parser.add_argument('config_name', type=str, help='configuration (scene) to optimize')
    parser.add_argument('input_name', type=str, help='output directories')

    args = parser.parse_args()
    loss_function = torch.nn.MSELoss()

    global config, outdir
    config = CONFIGS[args.config_name]

    opt_image = ru.image_to_torch(config['ref_texture'])
    ref_image = ru.image_to_torch(args.input_name)

    print(loss_function(ref_image, opt_image))
