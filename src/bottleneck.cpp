#include "bottleneck.h"
#include <torch/torch.h>
#include "conv_options.h"

BottleneckImpl::BottleneckImpl(int64_t inplanes, int64_t planes, int64_t stride_,
                     torch::nn::Sequential downsample_, int groups,
                     int base_width, bool _is_basic) {
  downsample = downsample_;
  stride = stride_;
  int width = int(planes * (base_width / 64.)) * groups;

  conv1 = torch::nn::Conv2d(
      conv_options(inplanes, width, 3, stride_, 1, groups, false));
  bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(width));
  conv2 = torch::nn::Conv2d(conv_options(width, width, 3, 1, 1, groups, false));
  bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(width));
  is_basic = _is_basic;
  if (!is_basic) {
    conv1 = torch::nn::Conv2d(conv_options(inplanes, width, 1, 1, 0, 1, false));
    conv2 = torch::nn::Conv2d(
        conv_options(width, width, 3, stride_, 1, groups, false));
    conv3 =
        torch::nn::Conv2d(conv_options(width, planes * 4, 1, 1, 0, 1, false));
    bn3 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes * 4));
  }

  register_module("conv1", conv1);
  register_module("bn1", bn1);
  register_module("conv2", conv2);
  register_module("bn2", bn2);
  if (!is_basic) {
    register_module("conv3", conv3);
    register_module("bn3", bn3);
  }

  if (!downsample->is_empty()) {
    register_module("downsample", downsample);
  }
}

torch::Tensor BottleneckImpl::forward(torch::Tensor x) {
  torch::Tensor residual = x.clone();

  x = conv1->forward(x);
  x = bn1->forward(x);
  x = torch::relu(x);

  x = conv2->forward(x);
  x = bn2->forward(x);

  if (!is_basic) {
    x = torch::relu(x);
    x = conv3->forward(x);
    x = bn3->forward(x);
  }

  if (!downsample->is_empty()) {
    residual = downsample->forward(residual);
  }

  x += residual;
  x = torch::relu(x);

  return x;
}