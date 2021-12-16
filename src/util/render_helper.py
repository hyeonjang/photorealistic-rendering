import os
import time
from os.path import join, dirname, realpath
from functools import partial


import enoki as ek
import numpy as np

from mitsuba.core import Bitmap, Struct, Thread, Log
from mitsuba.core.xml import load_file
from mitsuba.render import ImageBlock
from mitsuba.python.autodiff import render as mts_autodiff_render
from mitsuba.python.test.util import fresolver_append_path

@fresolver_append_path
def load_scene(filename, update_scene, **kwargs):
    """Prepares the file resolver and loads a Mitsuba scene from the given path."""
    fr = Thread.thread().file_resolver()
    here = os.path.dirname(__file__)
    fr.append(here)
    fr.append(os.path.join(here, filename))
    fr.append(os.path.dirname(filename))

    # kwargs.setdefault('color_mode', ColorMode.RGB)
    print(kwargs)
    scene = load_file(filename, update_scene, **kwargs)
    assert scene is not None
    return scene

def ensure_valid_values(params):
    # TODO: more principle way
    for k, v in params.items():
        if k.endswith('/rgb') or k.endswith('/bitmap') or k.endswith('/data_r')\
           or k.endswith('/data_g') or k.endswith('/data_b')\
           or ('alpha' in k and (k.endswith('/value') or k.endswith('/bitmap'))):
            params[k] = ek.clamp(ek.detach(v), 0, 1)
            # params[k] = type(v)(ek.clamp(ek.detach(v), 0, 1))
        elif k.endswith('/density') or ('texture3d' in k.lower() and k.endswith('/data')):
            # This is to avoid numerical precision issues
            params[k] = ek.clamp(ek.detach(v), 0., 6.)
            # params[k] = type(v)(ek.clamp(ek.detach(v), 0., 6.))
        else:
            continue
        ek.set_requires_gradient(params[k])


def save_info_csv(fname, **kwargs):
    headers = kwargs.keys()
    values = zip(*kwargs.values())
    with open(fname, 'w') as f:
        f.write(', '.join(headers) + '\n')
        for r in values:
            f.write(', '.join([str(v) for v in r]) + '\n')

def integrator_type_for_scene(scene, with_gradients, config=None):
    if config is not None and 'integrator' in config:
        name = config['integrator']
        return OptiXRenderer.EIntegratorType.__entries[name][0]

    has_volumes = scene.sensor().medium() is not None
    for shape in scene.shapes():
        has_volumes = has_volumes or shape.is_medium_transition()

    if has_volumes:
        return OptiXRenderer.ERadiativeBackpropVol if with_gradients else OptiXRenderer.EVolPath
    else:
        return OptiXRenderer.ERadiativeBackprop if with_gradients else OptiXRenderer.EPath

# -----------------------------------------------------------------------------


def compute_image_gradients(reference_b, current_b, loss_function,
                            cache=None, return_3d=False):
    """ Computes test inputs for radiative backropagation given a target image and current image.

    The input to radiative backprop is the gradient of the loss with respect
    to each image pixel.
    """
    assert current_b.pixel_format() == Bitmap.ERGB

    # Convert to differentiable Enoki arrays
    if cache is None:
        assert reference_b.pixel_format() == Bitmap.ERGB
        arr = np.array(reference_b, copy=False)
        ref = ek.Vector3fC(*[ek.FloatC(arr[:, :, i].flatten()) for i in range(3)])
        cache = {
            'reference': ref,
            'n_entries': ek.slices(ref),
            'index': 3 * ek.UInt32D.arange(ek.slices(ref)),
        }

    reference = cache['reference']
    arr = np.array(current_b, copy=False)
    current = ek.Vector3fD(*[ek.FloatD(arr[:, :, i].flatten()) for i in range(3)])

    ek.set_requires_gradient(current, True)
    l = loss_function(current, reference)
    ek.backward(l)

    grad3 = ek.gradient(current)
    if return_3d:
        grad_return = grad3
    else:
        # Flatten gradient vector
        grad_return = ek.FloatC.zero(len(grad3) * ek.slices(grad3))
        ek.scatter(grad_return, grad3[0], cache['index'])
        ek.scatter(grad_return, grad3[1], cache['index'] + 1)
        ek.scatter(grad_return, grad3[2], cache['index'] + 2)

    return grad_return, l, cache


def reference_differentiable_render(scene, reference_b, opt, loss_function,
                                    pass_count=10, spp=8, base_seed=12345,
                                    gradient_spp=None,
                                    output_dir=None):
    """Computes gradients of the loss with respect to some tracked parameters
    using automatic differentiation."""
    assert pass_count >= 1
    pixel_format = Bitmap.ERGB
    use_reattach_trick = gradient_spp is not None

    ref_flat = image_to_float_d(reference_b, pixel_format)
    gradients = {}

    for pass_i in range(pass_count):
        seed_i = base_seed + pass_i

        if use_reattach_trick:
            assert gradient_spp > 0
            seed2 = (pass_count + 1) + seed_i
            # Forward pass: high-quality image reconstruction, no gradients
            with opt.no_gradients():
                scene.sampler().seed(seed_i)
                image_block = mts_autodiff_render(scene, pixel_format=pixel_format, spp=spp)

            # Forward pass: low-spp gradient estimate
            scene.sampler().seed(seed2)
            gradients_block = mts_autodiff_render(
                scene, pixel_format=pixel_format, spp=gradient_spp)

            # Reattach trick
            block_values = image_to_float_d(image_block, pixel_format)
            gradient_block_values = image_to_float_d(gradients_block, pixel_format)
            ek.reattach(block_values, gradient_block_values)

        else:
            # Single render for both image values and gradient values (may be biased)
            scene.sampler().seed(seed_i)
            image_block = mts_autodiff_render(scene, pixel_format=pixel_format, spp=spp)
            block_values = image_to_float_d(image_block, pixel_format)

        if output_dir is not None:
            pass_fname = join(output_dir, 'autodiff_{:02d}.exr').format(pass_i)
            shape = (reference_b.width(), reference_b.height(), reference_b.channel_count())
            float_d_to_bitmap(block_values, shape, pixel_format).write(pass_fname)
            Log(EInfo, '[+] Saved pass debug image to: {}'.format(pass_fname))

        loss = loss_function(block_values, ref_flat)
        ek.backward(loss)

        for k, v in opt.params.items():
            if k not in gradients:
                gradients[k] = ek.gradient(v) / float(pass_count)
            else:
                gradients[k] += ek.gradient(v) / float(pass_count)
    return image_block, gradients


def finite_differences_render(scene, reference_b, opt, loss_function,
                              epsilon=1e-3,
                              pass_count=10, spp=8, base_seed=12345,
                              output_dir=None,
                              render_function=None, method_name='fd'):
    """Computes gradients of the loss with respect to some tracked parameters
    using finite differences."""
    assert pass_count >= 1
    pixel_format = Bitmap.ERGB

    # We don't need autodiff here
    for k, _ in opt.params.items():
        ek.set_requires_gradient(opt.params[k], False)


    render_function = render_function or mts_autodiff_render
    ref_flat = image_to_float_d(reference_b, pixel_format)
    gradients = {}

    render_counter = 0
    def finite_differences_single(update_param, epsilon):
        nonlocal render_counter, seed_i, pass_i
        loss_values = []
        for change_i, change in enumerate([-0.5 * epsilon, 0.5 * epsilon]):
            update_param(change)
            scene.sampler().seed(seed_i)  # Use the same seed for all change_i
            image_block = render_function(scene, pixel_format=pixel_format, spp=spp)
            block_values = image_to_float_d(image_block, pixel_format)
            loss = loss_function(block_values, ref_flat)
            loss_values.append(loss.numpy().squeeze())

            if output_dir is not None:
                pass_fname = join(output_dir, '{}_{:04d}_pass_{:02d}_{:d}.exr').format(method_name, render_counter, pass_i, change_i)
                shape = (reference_b.width(), reference_b.height(), reference_b.channel_count())
                float_d_to_bitmap(block_values, shape, pixel_format).write(pass_fname)
                # print('[+] Saved pass debug image to: {}'.format(pass_fname))
            render_counter += 1
        # Reset param to original value
        update_param(0)

        return (loss_values[1] - loss_values[0]) / epsilon

    def finite_differences(param_name, epsilon):
        v = opt.params[param_name]
        if isinstance(v, ek.FloatD):
            original_value = v.numpy().copy()
            result = np.zeros(len(v))
            for idx in range(len(v)):
                def update_param(diff, index):
                    value = original_value.copy()
                    value[index] += diff
                    opt.params[param_name] = ek.FloatD(value)
                result[idx] = finite_differences_single(
                    partial(update_param, index=idx), epsilon)
            return result

        elif isinstance(v, ek.Vector3fD):
            grad3 = np.zeros((3,))
            for k in range(3):
                assert len(
                    v[k]) == 1, "Not implemented yet: Vector3fD parameters with more than 1 entry"
                original_value = v.numpy()

                def update_param(diff):
                    diff3 = [0, 0, 0]
                    diff3[k] = diff
                    opt.params[param_name] = ek.Vector3fD(original_value + diff3)

                grad3[k] = finite_differences_single(update_param, epsilon)
            return grad3

        else:
            raise RuntimeError("Unsupported Enoki type in finite differences: {}".format(type(v)))

    # For every single parameter (including e.g. individual component of a color),
    # render the image twice to compute finite differences gradient.
    for pass_i in range(pass_count):
        seed_i = base_seed + pass_i
        scene.sampler().seed(seed_i)

        for k, _ in opt.params.items():
            # Compute gradients with finite differences
            grad_k = finite_differences(k, epsilon) / float(pass_count)

            if k not in gradients:
                gradients[k] = grad_k
            else:
                gradients[k] += grad_k

    # Convert gradients to Enoki types for consistency with other methods
    for k, v in opt.params.items():
        tp = type(ek.detach(v))
        gradients[k] = tp(gradients[k])

    return gradients

# =============================================================================================== 
# added
# =============================================================================================== 
def render_reference(scene_path, output, spp=1024):
    scene = load_scene(scene_path, parameters=[('spp', str(spp))])
    scene.integrator().render(scene, vectorize=False)
    film = scene.film()
    film.set_destination_file(output, 32)
    film.develop()
    print('[+] Rendered reference image at {} spp: {}'.format(spp, output))


def load_reference(filename):
    ref = Bitmap(filename).convert(Bitmap.PixelFormat.RGB, Struct.Type.Float32, False)
    # return linearize_image(ref), ref
    return ref


# def linearize_image(image):
#     if isinstance(image, ImageBlock):
#         X, Y, Z, _, W = image.bitmap_d()
#     else:
#         assert isinstance(image, Bitmap)
#         arr = np.array(image, copy=False)
#         channels = [ek.FloatD(arr[:, :, k].flatten()) for k in range(image.channel_count())]

#         if image.pixel_format() == Bitmap.ERGB:
#             # No conversion to do
#             return ek.Vector3fD(*channels)

#         assert len(channels) == 5
#         X, Y, Z, _, W = channels

#     # Apply normalization weight
#     W = ek.select(ek.eq(W, 0), 0, ek.rcp(W))
#     X *= W
#     Y *= W
#     Z *= W
#     # XYZ to RGB conversion
#     return ek.Vector3fD(
#         3.240479 * X + -1.537150 * Y + -0.498535 * Z,
#         -0.969256 * X + 1.875991 * Y + 0.041556 * Z,
#         0.055648 * X + -0.204043 * Y + 1.057311 * Z
#     )

def refresh_requires_gradient(params):
    """Note: this is a temporary workaround that will not be needed in the future."""
    print(params)
    for k, _ in params.items():
        ek.set_requires_gradient(params[k], True)
        parent = params.parent(k)
        if parent is not None:
            parent.parameters_changed()

def refresh_requires_gradient(params):
    """Note: this is a temporary workaround that will not be needed in the future."""
    print(params)
    for k, _ in params.items():
        ek.set_requires_gradient(params[k], True)
        parent = params.parent(k)
        if parent is not None:
            parent.parameters_changed()

def optimize_scene(scene, reference_image, optimizer, loss_function, regularizer=None,
                   spp=16, max_iterations=100,
                   output_dir=None, write_intermediate=False):

    from mitsuba.python.util import traverse
    from mitsuba.python.autodiff import Adam, render, write_bitmap

    if write_intermediate:
        crop_size = scene.sensors()[0].film().crop_size()

    # Optimization loop
    for i in range(max_iterations):
        t0 = time.time()
        rendered = render(scene, optimizer=optimizer, spp=spp)

        # L2 loss
        loss = loss_function(reference_image, rendered)

        # add regularization value

        # Backpropagate gradients of the loss w.r.t. inputs and take a descent step
        ek.backward(loss)
        optimizer.step()

        # Save current image & print current loss
        if write_intermediate:
            write_bitmap(join(output_dir, f"out_{i:04d}.exr"), rendered, crop_size)
        loss_v = loss.numpy()[0]
        print('[{:03d}] Loss: {:.6f}, took {:.2f}s'.format(i, loss_v, time.time() - t0), end='\r')

    # Final values
    print('\nFinal loss: {:.6f}'.format(loss_v))
    print('Final parameter values:')
    for k, v in optimizer.params.items():
        print('- {}: {}'.format(k, v))

# ===========================================================================
# Mitsuba render
# ===========================================================================
# render function
def render(scene, params=None, **kwargs):
    from mitsuba.core import Float
    from mitsuba.python.autodiff import render
    # Delayed import of PyTorch dependency
    ns = globals()
    if 'render_torch_helper' in ns:
        render_torch = ns['render_torch_helper']
    else:
        import torch

        class Render(torch.autograd.Function):
            @staticmethod
            def forward(ctx, scene, params, *args):
                try:
                    assert len(args) % 2 == 0
                    args = dict(zip(args[0::2], args[1::2]))

                    spp = None
                    sensor_index = 0
                    unbiased = True
                    malloc_trim = False

                    ctx.inputs = [None, None]
                    ctx.templs = [None, None]

                    for k, v in args.items():
                        if k == 'spp':
                            spp = v
                        elif k == 'sensor_index':
                            sensor_index = v
                        elif k == 'unbiased':
                            unbiased = v
                        elif k == 'malloc_trim':
                            malloc_trim = v
                        elif params is not None:
                            temp = v.shape
                            params[k] = type(params[k])(v.flatten())


                            ctx.inputs.append(None)
                            ctx.inputs.append(params[k] if v.requires_grad else None)

                            ctx.templs.append(None)
                            ctx.templs.append(temp if v.requires_grad else None)
                            continue

                        ctx.inputs.append(None)
                        ctx.inputs.append(None)

                        ctx.templs.append(None)
                        ctx.templs.append(None)

                    if type(spp) is not tuple:
                        spp = (spp, spp)

                    result = None
                    ctx.malloc_trim = malloc_trim

                    if ctx.malloc_trim:
                        torch.cuda.empty_cache()

                    if params is not None:
                        params.update()

                        if unbiased:
                            result = render(scene, spp=spp[0], sensor_index=sensor_index).torch()

                    for v in ctx.inputs:
                        if v is not None:
                            ek.set_requires_gradient(v)

                    ctx.output = render(scene, spp=spp[1], sensor_index=sensor_index)
                    if result is None:
                        result = ctx.output.torch()

                    if ctx.malloc_trim:
                        ek.cuda_malloc_trim()
                    return result
                except Exception as e:
                    print("render_torch(): critical exception during "
                          "forward pass: %s" % str(e))
                    raise e

            @staticmethod
            def backward(ctx, grad_output):
                try:
                    ek.set_gradient(ctx.output, ek.cuda.Float32(grad_output))
                    Float.backward()

                    result = tuple(ek.gradient(ctx.inputs[i]).torch().reshape(ctx.templs[i]) if ctx.inputs[i] is not None
                                   else None
                                   for i, _ in enumerate(ctx.inputs))

                    del ctx.output 
                    del ctx.inputs
                    if ctx.malloc_trim:
                        ek.cuda_malloc_trim()
                    return result
                except Exception as e:
                    print("render_torch(): critical exception during "
                          "backward pass: %s" % str(e))
                    raise e

        render_torch = Render.apply
        ns['render_torch_helper'] = render_torch

    result = render_torch(scene, params,
                          *[num for elem in kwargs.items() for num in elem])

    sensor_index = 0 if 'sensor_index' not in kwargs else kwargs['sensor_index']
    crop_size = scene.sensors()[sensor_index].film().crop_size()
    return result.reshape(crop_size[1], crop_size[0], -1)