import os
import numpy as np

import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import Float, Thread
from mitsuba.core import xml
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render, write_bitmap, Adam

# Load example scene
Thread.thread().file_resolver().append('./data/scene/veach_mi')

# output dir
out_dir = "result-veach"
os.makedirs(out_dir, exist_ok=True)

opt_scene = xml.load_file('data/scene/veach_mi/mi.xml') # construct_scene("museum.exr", "texture/256x256.png")

# Find differentiable scene parameters
# ref_scene = xml.load_file("data/bunny.xml")
params = traverse(opt_scene)
print(params)

# Discard all parameters except for one we want to differentiate
params.keep([
    'Sphere.emitter.radiance.value',
    'Sphere_1.emitter.radiance.value',
    'Sphere_2.emitter.radiance.value',
    'Sphere_3.emitter.radiance.value',

    'OBJMesh.bsdf.specular_reflectance.value',
    'OBJMesh_1.bsdf.specular_reflectance.value',
    'OBJMesh_2.bsdf.specular_reflectance.value',
    'OBJMesh_3.bsdf.specular_reflectance.value',
])
# Render a reference image (no deriva
# tives used yet)
image_ref = render(opt_scene, spp=16)
crop_size = opt_scene.sensors()[0].film().crop_size()
write_bitmap(os.path.join(out_dir, 'out_ref.png'), image_ref, crop_size)

# params['OBJMesh.bsdf.specular_reflectance.value'] = Float(0.0)
# params['OBJMesh_1.bsdf.specular_reflectance.value'] = Float(0.0) 
# params['OBJMesh_2.bsdf.specular_reflectance.value'] = Float(0.0)
# params['OBJMesh_3.bsdf.specular_reflectance.value'] = Float(0.0)
# params.update()

# Change to a dark color to diffuse texture

# def compute_groundtruth(make_scene, integrator, spp, passes, epsilon):
def compute_groundtruth(scene, param, param_ref, passes, epsilon):
    """Render groundtruth radiance and gradient image using finite difference"""
    from mitsuba.python.autodiff import render

    param['OBJMesh.bsdf.specular_reflectance.value'] = params_ref['OBJMesh.bsdf.specular_reflectance.value']
    param['OBJMesh_1.bsdf.specular_reflectance.value'] = params_ref['OBJMesh_1.bsdf.specular_reflectance.value']
    param['OBJMesh_2.bsdf.specular_reflectance.value'] = params_ref['OBJMesh_2.bsdf.specular_reflectance.value']
    param['OBJMesh_3.bsdf.specular_reflectance.value'] = params_ref['OBJMesh_3.bsdf.specular_reflectance.value']
    def render_offset(offset):
        param['OBJMesh.bsdf.specular_reflectance.value'] = 0.9
        param['OBJMesh_1.bsdf.specular_reflectance.value'] = 0.9
        param['OBJMesh_2.bsdf.specular_reflectance.value'] = 0.9
        param['OBJMesh_3.bsdf.specular_reflectance.value'] = 0.9
        fsize = scene.sensors()[0].film().size()

        values = render(scene)
        for i in range(passes-1):
            values += render(scene)
        values /= passes

        return values.numpy().reshape(fsize[1], fsize[0], -1)

    gradient = (render_offset(epsilon) - render_offset(-epsilon)) / (2.0 * ek.norm(epsilon))
    image = render_offset(0.0)

    return image, gradient[:, :, [0]]

def write_gradient_image(grad, name):
    """Convert signed floats to blue/red gradient exr image"""
    from mitsuba.core import Bitmap

    convert_to_rgb = False

    if convert_to_rgb:
        # Compute RGB channels for .exr image (no grad = black)
        grad_R = grad.copy()
        grad_R[grad_R < 0] = 0.0
        # print(grad_R<0)
        # print(grad_R)
        grad_B = grad.copy()
        grad_B[grad_B < 0] = 0.0
        # grad_B *= -1.0
        # grad_B *= 0.0
        grad_G = grad.copy() * 0.0
        # grad_G[grad_G < 0] *= -1.0

        grad_np = np.concatenate((grad_R, grad_G, grad_B), axis=2)
    else:
        grad_np = np.concatenate((grad, grad, grad), axis=2)

    print('Writing', name + ".exr")
    Bitmap(grad_np).write(name + ".exr")

def make_scene():
    params['OBJMesh.bsdf.specular_reflectance.value'] = params['OBJMesh.bsdf.specular_reflectance.value'] + Float(0.01)

params_ref = {}
params_ref['OBJMesh.bsdf.specular_reflectance.value'] = Float(params['OBJMesh.bsdf.specular_reflectance.value'])
params_ref['OBJMesh_1.bsdf.specular_reflectance.value'] = Float(params['OBJMesh_1.bsdf.specular_reflectance.value'])
params_ref['OBJMesh_2.bsdf.specular_reflectance.value'] = Float(params['OBJMesh_2.bsdf.specular_reflectance.value'])
params_ref['OBJMesh_3.bsdf.specular_reflectance.value'] = Float(params['OBJMesh_3.bsdf.specular_reflectance.value'])

ref_image, ref_grad = compute_groundtruth(opt_scene, params, params_ref, 10, 0.002)
print(ref_grad.shape)
scale = np.abs(ref_grad).max()
write_gradient_image(ref_grad, "ref")

ek.set_requires_gradient(params['OBJMesh.bsdf.specular_reflectance.value'])
ek.set_requires_gradient(params['OBJMesh_1.bsdf.specular_reflectance.value'])
ek.set_requires_gradient(params['OBJMesh_2.bsdf.specular_reflectance.value'])
ek.set_requires_gradient(params['OBJMesh_3.bsdf.specular_reflectance.value'])

# ek.set_requires_gradient(params['Sphere.emitter.radiance.value'])
# ek.set_requires_gradient(params['Sphere_1.emitter.radiance.value'])
# ek.set_requires_gradient(params['Sphere_2.emitter.radiance.value'])
# ek.set_requires_gradient(params['Sphere_3.emitter.radiance.value'])

image = render(opt_scene)
ek.set_gradient(params['OBJMesh.bsdf.specular_reflectance.value'], 0.9, backward=False)
ek.set_gradient(params['OBJMesh_1.bsdf.specular_reflectance.value'], 0.9, backward=False)
ek.set_gradient(params['OBJMesh_2.bsdf.specular_reflectance.value'], 0.9, backward=False)
ek.set_gradient(params['OBJMesh_3.bsdf.specular_reflectance.value'], 0.9, backward=False)

# ek.set_gradient(params['Sphere.emitter.radiance.value'], [700, 700, 700], backward=False)
# ek.set_gradient(params['Sphere_1.emitter.radiance.value'], [700, 700, 700], backward=False)
# ek.set_gradient(params['Sphere_2.emitter.radiance.value'], [700, 700, 700], backward=False)
# ek.set_gradient(params['Sphere_3.emitter.radiance.value'], [700, 700, 700], backward=False)

Float.forward()
image_grad = ek.gradient(image)

crop_size = opt_scene.sensors()[0].film().crop_size()
fname = 'out.png'
write_bitmap(fname, image, crop_size)
print(image_grad)
write_gradient_image(image_grad.numpy().reshape(crop_size[1], crop_size[0], -1)[:,:,[1]], "forward")
exit()

def render_gradient(scene, passes, param):

    fsize = scene.sensors()[0].film().size()

    diff_param = Float(0.0)
    ek.set_requires_gradient(diff_param)

    img = render(scene)
    ek.forward(img)
    image = ek.gradient(img)

    write_bitmap("tmp.png", image, fsize)



render_gradient(opt_scene, 1, params)
exit()


# Construct an Adam optimizer that will adjust the parameters 'params'
opt = Adam(params, lr=.02)

for it in range(101):
    # Perform a differentiable rendering of the scene
    image = render(opt_scene, optimizer=opt, unbiased=True, spp=8)

    if it%50==0:
        write_bitmap(os.path.join(out_dir, 'out_%03i.png') % it, image, crop_size)

    # Objective: MSE between 'image' and 'image_ref'
    ob_val = ek.hsum(ek.sqr(image - image_ref)) / len(image)

    # Back-propagate errors to input parameters
    ek.backward(ob_val)

    # Optimizer: take a gradient step
    opt.step()

    # Compare iterate against ground-truth value
    print('Iteration %03i: error=%g' % (it, ob_val[0]))