#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <omp.h>

#include <future>

using namespace torch::indexing;

torch::Tensor laplacian_smooth(
  torch::Tensor image, 
  torch::Tensor gradient, 
  torch::Scalar step_size) {

  const int padding = 3;

  float array[] = {
    0.1070, 0.1131, 0.1070,
    0.1131, 0.1196, 0.1131,
    0.1070, 0.1131, 0.1070,
  };
  torch::Tensor gk_2d = torch::from_blob(array, {3, 3});

  torch::Tensor output = torch::zeros_like(image);

  // initialize: add padding
  torch::Tensor pad_image = torch::ones({
    image.size(0)+2, 
    image.size(1)+2, 
    image.size(2)}
    );
  torch::Tensor pad_g_img = pad_image.clone();

  // initialize: real value
  pad_image.index_put_({Slice(1, -1), Slice(1, -1), Slice()}, image);
  pad_g_img.index_put_({Slice(1, -1), Slice(1, -1), Slice()}, gradient);

  // iteration
  for(int d=0; d<image.size(2); d++)

    #pragma omp parallel for num_threads(omp_get_max_threads())
    for(int h=0; h<image.size(1); h++)
      for(int w=0; w<image.size(0); w++) {

        auto kern = gk_2d.mm(pad_image.index({Slice(w, w+3), Slice(h, h+3), d}));
        auto diag = torch::diag(torch::diag(kern));
        auto weig = torch::linalg::inv(diag).mm(kern);
        auto lapl = weig-diag;
        auto I_L = torch::eye(kern.size(0)) + lapl.mul_(step_size);
        output.index_put_(
          {w, h, d}, 
          I_L.inverse().mm(pad_g_img.index({Slice(w, w+3), Slice(h, h+3), d})).sum()
          );
      }
  return output;
}

TORCH_LIBRARY(image, m) {
  m.def("laplacian_smooth", &laplacian_smooth);
}