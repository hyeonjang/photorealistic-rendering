import mitsuba

import enoki as ek

from mitsuba.core import Int32, UInt32, Float32, Vector2i, Vector2u, Vector2f, Vector3i, Vector3u, Vector3f
from mitsuba.core import Bitmap
from mitsuba.render import ImageBlock

import numpy as np

def ravel_float(buf, dim = 3):
    idx = dim * UInt32.arange(ek.slices(buf) // dim)

    if dim == 2:
        return Vector2f(ek.gather(buf, idx), ek.gather(buf, idx + 1))
    elif dim == 3:
        return Vector3f(ek.gather(buf, idx), ek.gather(buf, idx + 1), ek.gather(buf, idx + 2))

def ravel_uint(buf, dim = 3):
    idx = dim * UInt32.arange(ek.slices(buf) // dim)

    if dim == 2:
        return Vector2u(ek.gather(buf, idx), ek.gather(buf, idx+1))
    elif dim == 3:
        return Vector3u(ek.gather(buf, idx), ek.gather(buf, idx + 1), ek.gather(buf, idx + 2))

def ravel_int(buf, dim = 3):
    idx = dim * UInt32.arange(ek.slices(buf) // dim)

    if dim == 2:
        return Vector2i()
    elif dim == 3:
        return Vector3i(ek.cuda_autodiff.Int32(ek.gather(buf, idx)), ek.cuda_autodiff.Int32(ek.gather(buf, idx+1)), ek.cuda_autodiff.Int32(ek.gather(buf, idx+2)))

def ravel(buf, dim = 3):

    if isinstance(buf, Int32):
        return ravel_int(buf)

    if isinstance(buf, UInt32):
        return ravel_uint(buf)

    if isinstance(buf, Float32):
        return ravel_float(buf)

def unravel(source, target, dim = 3):
    from mitsuba.core import UInt32

    idx = UInt32.arange(ek.slices(source))
    for i in range(dim):
        ek.scatter(target, source[i], dim * idx + i)

def image_to_torch(path):
    import torch
    from mitsuba.core import Bitmap, Struct

    image = np.asarray(Bitmap(path).convert(Bitmap.PixelFormat.RGB, Struct.Type.Float32, False))
    return torch.from_numpy(image)

def write_image(path, image, size=None):
    from mitsuba.python.autodiff import write_bitmap

    if size is not None:
        write_bitmap(path, image, (size[0], size[1]))
    else:
        write_bitmap(path, image, (image.shape[0], image.shape[1]))
