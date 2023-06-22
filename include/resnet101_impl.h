#ifndef RESNET_101_IMPL_H
#define RESNET_101_IMPL_H

#include <torch/nn/pimpl.h>
#include <torch/torch.h>

class ResNet101Impl : public torch::nn::Module {
 public:
  ResNet101Impl() {}
  ResNet101Impl(std::vector<int> layers, int num_classes = 1000,
                std::string model_type = "resnet18", int groups = 1,
                int width_per_group = 64);
  torch::Tensor forward(torch::Tensor x);
  std::vector<torch::Tensor> features(torch::Tensor x);
  torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks,
                                    int64_t stride = 1);

 private:
  int expansion = 1;
  bool is_basic = true;
  int64_t inplanes = 64;
  int groups = 1;
  int base_width = 64;
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