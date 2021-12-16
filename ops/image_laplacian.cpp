#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <omp.h>

#include <future>

using namespace torch::indexing;

void temp() {

}

torch::Tensor laplacian_smooth(torch::Tensor image, torch::Tensor gradient) {

  const int padding = 3;

  std::cout << "print somthing" << std::endl;
  std::cout << image.sizes() << std::endl;

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

  omp_set_num_threads(32);
  std::cout << omp_get_thread_num() << std::endl;

  // iteration
  #pragma omp parallel for
  for(int z=0; z<image.size(2); z++)
    for(int x=0; x<image.size(1); x++)
      for(int y=0; y<image.size(0); y++) {

        auto k = pad_image.index({Slice(y, y+3), Slice(x, x+3), z});
        auto d = torch::diag(torch::diag(k));
        auto w = torch::linalg::inv(d).mm(k);
        auto l = w-d;
        auto I_L = torch::eye(k.size(0)) + l;
        output.index_put_(
          {y, x, z}, 
          I_L.inverse().mm(pad_g_img.index({Slice(y, y+3), Slice(x, x+3), z})).sum()
          );
      }

  return output;
}

TORCH_LIBRARY(image, m) {
  m.def("laplacian_smooth", &laplacian_smooth);
}