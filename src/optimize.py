import os
import time
import numpy as np
import argparse

import torch

import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import Bitmap
from mitsuba.python.util import traverse

from util import loss
from util import optimizer
from util import render_helper as rh
from util import render_util as ru

from config import CONFIGS, OUTPUT_DIR

cuda = torch.device('cuda')

def make_output_dir(config, opt_name):
    # output dir
    config_name = config['name']

    if opt_name is not None:
        config_name += '_' + opt_name

    ref_dir = os.path.join(OUTPUT_DIR, config_name)
    out_dir = os.path.join(OUTPUT_DIR, config_name, 'ad')
    tex_dir = os.path.join(OUTPUT_DIR, config_name, 'tex')
    env_dir = os.path.join(OUTPUT_DIR, config_name, 'env')
    
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tex_dir, exist_ok=True)
    os.makedirs(env_dir, exist_ok=True)

    return {"ref":ref_dir, "out":out_dir, "tex":tex_dir, "env":env_dir}

def make_optimizer():
    if config['optimizer'] == 'large-steps':
        return optimizer.UniformAdam
    elif config['optimizer'] == 'adam':
        return torch.optim.Adam

def make_reference():
    '''
    Reference image and texture construction
    '''
    # Render reference image, if missing
    # if not os.path.isfile(config['ref_image']):
    scene = rh.load_scene(config['ref_scene'], False, parameters=config['ref_scene_params'])
    ref_image = rh.render(scene, spp=64)
    ru.write_image(config['ref_image'], ref_image)
    print('[+] Saved reference image: {}'.format(config['ref_image']))

    # call reference image
    ref_texture = None
    if config['ref_texture'] is not None:
        ref_texture = ru.image_to_torch(config['ref_texture']).to(device=cuda)

    ref_envmap = None
    if config['ref_envmap'] is not None:
        ref_envmap = ru.image_to_torch(config['ref_envmap']).to(device=cuda)

    return ref_image, ref_texture, ref_envmap

def optimize_ad():

    # =======================================================
    # construct scene to optimize
    # =======================================================
    scene = rh.load_scene(config['opt_scene'], False, parameters=config.get('scene_params', []))
    params_names = config['params']

    # set optimization target: texture
    texture = ru.image_to_torch('./data/texture/512x512.png')
    prop_torch = {}
    for name in params_names:
        prop_torch[name] = texture.clone().to(device=cuda).requires_grad_()

    # preprocess function for paramters
    params = traverse(scene)
    print(params)
    params.keep(params_names)
    params_torch = params.torch()

    # ========================================================
    # Reference image and texture construction
    # ========================================================
    # Render reference image, if missing
    ref_image, ref_texture, ref_envmap = make_reference()
    params_ref = {}

    if ref_texture is not None:
        params_ref[params_names[0]] = ref_texture
    # if ref_envmap is not None:
    #     params_ref[params_names[0]] = ref_envmap
    
    # ========================================================
    # trainning option creation 
    # ========================================================
    # Create loss function 
    loss_function = torch.nn.MSELoss()

    # ========================================================
    # optimizer (! comparision)
    learning_rate = config.get('learning_rate', .1)
    optimize = make_optimizer()
    opt = optimize(prop_torch.values(), lr=learning_rate)

    # rendering and optimize option
    spp = config.get('forward_spp', 32)
    max_iterations = config.get('max_iterations', 50)

    # record
    f = open(os.path.join(outdir['tex'], "record.txt"), 'w')

    # Optimization loop
    for i in range(max_iterations):
        t0 = time.time()
        opt.zero_grad()
        
        for name in params_names:
            params_torch[name] = torch.sigmoid(prop_torch[name])
            tex0 = mitsuba.core.Float(params_torch[name].flatten())
            tex1 = ru.ravel_float(tex0)
            ru.unravel(tex1, params[name])
            params.set_dirty(name)
            params.update()

        rendered = rh.render(scene, params=params, unbiased=True, spp=spp, **params_torch)

        # L2 loss
        image_loss = loss_function(rendered, ref_image)
        
        for name in params_names:
            param_loss = loss_function(params_torch[name], params_ref[name])
            f.write(f"{name}: {param_loss}\n")
            print(f'[{i:03d}] {name}_loss: {param_loss:.6f}, took {time.time()-t0:.2f}s')

        # Backpropagate gradients of the loss w.r.t. inputs and take a descent step
        image_loss.backward()
        opt.step()

        print('[{:03d}] Loss: {:.6f}, took {:.2f}s'.format(i, image_loss, time.time() - t0))
        ru.write_image(os.path.join(outdir['out'], f"out_{i:04d}.exr"), rendered)
        ru.write_image(os.path.join(outdir['tex'], f"tex_{i:04d}.exr"), params[params_names[0]], (512,512))
        # ru.write_image(os.path.join(outdir['env'], f"env_{i:04d}.exr"), params[params_names[1]], (512,512))

    f.close()

if __name__ == '__main__':  

    parser = argparse.ArgumentParser(prog='test_radiative_backprop.py')
    parser.add_argument('config_name', type=str, help='configuration (scene) to optimize')
    parser.add_argument('--output_dir', type=str, help='output directories')
    args = parser.parse_args()

    global config, outdir
    config = CONFIGS[args.config_name]
    outdir = make_output_dir(config, args.output_dir)
    print(outdir)

    optimize_ad()