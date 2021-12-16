import torch
torch.ops.load_library("build/ops/libimage_laplacian.so")
import cv2

if __name__ == '__main__':

    image = torch.tensor([
        [[1, 2, 1, 1, 1],
         [1, 2, 1, 1, 1],
         [1, 2, 1, 1, 1],
         [1, 3, 1, 0, 1]],
        [[1, 2, 1, 1, 1],
         [1, 2, 1, 1, 1],
         [1, 2, 1, 1, 1],
         [1, 3, 1, 0, 1]],
        [[1, 2, 1, 1, 1],
         [1, 2, 1, 1, 1],
         [1, 2, 1, 1, 1],
         [1, 3, 1, 0, 1]]])

    g_img = torch.tensor([
        [1, 2, 1],
        [1, 2, 1],
        [1, 3, 1],
    ])

    kernel = torch.from_numpy(cv2.getGaussianKernel(3,3, cv2.CV_32F))

    print(kernel[:, 0].shape)
    print(torch.outer(kernel[:, 0], kernel[:, 0]))


    torch.ops.image.laplacian_smooth(image, g_img, 1)