import torch
torch.ops.load_library("build/ops/libimage_laplacian.so")

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

    torch.ops.image.laplacian_smooth(image, g_img)