#ifndef RESNET_101_IMPL_H
#define RESNET_101_IMPL_H

#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/pooling.h>
#include <torch/nn/pimpl.h>
#include <torch/torch.h>
#include <cstdint>
#include <vector>

struct TORCH_API ResNet101Options {
  ResNet101Options(std::vector<int> layers, int num_classes);

  TORCH_ARG(std::vector<int>, layers);

  TORCH_ARG(int, expansion);

  TORCH_ARG(bool, is_basic);

  TORCH_ARG(int64_t, inplanes);

  TORCH_ARG(int, base_width);

  TORCH_ARG(int, num_classes);

  TORCH_ARG(int, num_channels);

  TORCH_ARG(std::string, model_type);

  TORCH_ARG(int, groups);

  TORCH_ARG(int, width_per_group);
};

class ResNet101Impl : public torch::nn::Module {
 public:
  // ResNet101Impl() {}
  ResNet101Impl(const ResNet101Options& options_);
  torch::Tensor forward(torch::Tensor x);
  std::vector<torch::Tensor> features(torch::Tensor x);
  torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks,
                                    int64_t stride = 1);

 private:
  // int expansion = 1;  // neck::expansion
  // bool is_basic = true;
  // int64_t inplanes = 64;
  // int groups = 1;
  // int base_width = 64;
  ResNet101Options options;
  torch::nn::Conv2d conv1 = nullptr;
  torch::nn::BatchNorm2d bn1 = nullptr;
  torch::nn::Sequential layer1 = nullptr;
  torch::nn::Sequential layer2 = nullptr;
  torch::nn::Sequential layer3 = nullptr;
  torch::nn::Sequential layer4 = nullptr;
  torch::nn::Linear fc = nullptr;
};

TORCH_MODULE(ResNet101);

#endif  // RESNET_101_IMPL_H