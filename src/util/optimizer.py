import math
import time
import numpy as np
import torch
from torch import Tensor
torch.ops.load_library("build/ops/libimage_laplacian.so")
from typing import List

import numpy as np
import cv2

def make_kernel():
    kernel1d = cv2.getGaussianKernel(3, 3)
    kernel2d = np.outer(kernel1d, kernel1d.transpose())
    return kernel2d

def laplacian_operator(k2d, type):

    d = np.diag(np.diag(k2d))
    w = np.linalg.inv(d) @ k2d

    l = None
    if type == "unnormalize":
        l = w-d
    elif type == "normalize":
        l = (np.sqrt(np.linalg.inv(d))@w@np.sqrt(np.linalg.inv(d)))

    return l

def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        step_size = lr / bias_correction1

        with torch.no_grad():
            kernel1d = cv2.getGaussianKernel(3, 3)
            kernel2d = np.outer(kernel1d, kernel1d.transpose())
            k = kernel2d

            d = np.diag(np.diag(k))
            w = np.linalg.inv(d) @ k

            # todo part
            i = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])

            def convolve2d(image, gradi):
                """
                This function which takes an image and a kernel and returns the convolution of them.
                :param image: a numpy array of size [image_height, image_width].
                :param kernel: a numpy array of size [kernel_height, kernel_width].
                :return: a numpy array of size [image_height, image_width] (convolution output).
                """
                padding = 3

                # Flip the kernel
                # kernel = np.flipud(np.fliplr(kernel))
                gaussian = cv2.getGaussianKernel(3, 3, cv2.CV_32F)
                kernel = np.outer(gaussian, gaussian.transpose())
                # convolution output
                output = torch.zeros_like(image)

                # Add zero padding to the input image
                image_padded = torch.zeros((image.shape[0] + padding-1, image.shape[1] + padding-1, image.shape[2]))
                gradi_padded = torch.zeros((image.shape[0] + padding-1, image.shape[1] + padding-1, image.shape[2]))
                image_padded[1:-1, 1:-1, :] = image
                gradi_padded[1:-1, 1:-1, :] = gradi

                # Loop over every pixel of the image
                for z in range(image.shape[2]):
                    for x in range(image.shape[1]):
                        for y in range(image.shape[0]):
                            # element-wise multiplication of the kernel and the image
                            k = torch.from_numpy(kernel) @ image_padded[y:y+padding, x:x+padding, z]
                            d = torch.diag(torch.diag(k))
                            w = torch.linalg.pinv(d) @ k
                            l = w-d 
                            I_L = torch.eye(k.shape[0]) + l
                            output[y, x, z] = (I_L.inverse() @ gradi_padded[y:y+padding, x:x+padding, z]).sum()
                return output

            # l_p = i + step_size*(w-d)

            z = time.time()
            # g = cv2.filter2D(grad.detach().cpu().numpy(), -1, l_p)
            # g = torch.from_numpy(g).to(device='cuda')
            g = torch.ops.image.laplacian_smooth(param, grad)
            # g = convolve2d(param.detach(), grad.detach())
            print(time.time()-z)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(g, alpha=1 - beta1) # m1
        exp_avg_sq.mul_(beta2).addcmul_(g, g.conj(), value=1 - beta2) # m2
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        # x(u) = (I + \lambda)_inv u
        param.addcdiv_(exp_avg, denom, value=-step_size)

class UniformAdam(torch.optim.Optimizer):
    '''
    Uniform Adam from Baptiste Nicolet, "Larget Stes in Inverse Rendering of Geometry" 2021 SIGGRAPH ASIA
    actually not uniform adam cases
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(UniformAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(UniformAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0 
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])


            adam(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   amsgrad=group['amsgrad'],
                   beta1=beta1,
                   beta2=beta2,
                   lr=group['lr'],
                   weight_decay=group['weight_decay'],
                   eps=group['eps'])
        return loss