#ifndef BLOCK_H
#define BLOCK_H

#include <torch/torch.h>

class BlockImpl : public torch::nn::Module {
 public:
  BlockImpl(int64_t inplanes, int64_t planes, int64_t stride_ = 1,
            torch::nn::Sequential downsample_ = nullptr, int groups = 1,
            int base_width = 64, bool is_basic = true);
  torch::Tensor forward(torch::Tensor x);
  torch::nn::Sequential downsample{nullptr};

 private:
  bool is_basic = true;
  int64_t stride = 1;
  torch::nn::Conv2d conv1{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr};
  torch::nn::Conv2d conv2{nullptr};
  torch::nn::BatchNorm2d bn2{nullptr};
  torch::nn::Conv2d conv3{nullptr};
  torch::nn::BatchNorm2d bn3{nullptr};
};
TORCH_MODULE(Block);

#endif  // BLOCK_H