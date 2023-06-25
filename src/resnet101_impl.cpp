#include "resnet101_impl.h"
#include <ATen/Functions.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/pooling.h>
#include <torch/nn/options/pooling.h>
#include <torch/nn/pimpl.h>
#include <torch/serialize.h>
#include <torch/torch.h>
#include <string>
#include <vector>
#include "bottleneck.h"
#include "conv_options.h"

ResNet101Options::ResNet101Options(std::vector<int> layers, int num_classes)
    : layers_(layers), num_classes_(num_classes) {
  expansion_ = 1;
  is_basic_ = true;
  inplanes_ = 64;
  groups_ = 1;
  base_width_ = 64;
}

torch::nn::Sequential ResNet101Impl::_make_layer(int64_t planes, int64_t blocks,
                                                 int64_t stride) {
  torch::nn::Sequential downsample;
  if (stride != 1 || options.inplanes() != planes * options.expansion()) {
    downsample = torch::nn::Sequential(
        torch::nn::Conv2d(conv_options(options.inplanes(),
                                       planes * options.expansion(), 1, stride,
                                       0, 1, false)),
        torch::nn::BatchNorm2d(planes * options.expansion()));
  }
  torch::nn::Sequential layers;
  // TODO 用ResBottleneck替换Bottleneck，提供可供选择的Bottleneck
  layers->push_back(Bottleneck(options.inplanes(), planes, stride, downsample,
                          options.groups(), options.base_width(),
                          options.is_basic()));
  options.inplanes() = planes * options.expansion();
  for (int64_t i = 1; i < blocks; i++) {
    layers->push_back(Bottleneck(options.inplanes(), planes, 1,
                            torch::nn::Sequential(), options.groups(),
                            options.base_width(), options.is_basic()));
  }

  return layers;
}

// ResNet101Impl::ResNet101Impl(std::vector<int> layers, int num_classes,
//                              int num_channels, std::string model_type,
//                              int _groups, int _width_per_group) {
ResNet101Impl::ResNet101Impl(const ResNet101Options& options_)
    : options(options_) {
  if (options.model_type() != "resnet18" &&
      options.model_type() != "resnet34") {
    options.expansion() = 4;
    options.is_basic() = false;
  }
  conv1 = torch::nn::Conv2d(
      conv_options(options.num_channels(), 64, 7, 2, 3, 1, false));
  bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
  layer1 = torch::nn::Sequential(_make_layer(64, options.layers()[0]));
  layer2 = torch::nn::Sequential(_make_layer(128, options.layers()[1], 2));
  layer3 = torch::nn::Sequential(_make_layer(256, options.layers()[2], 2));
  layer4 = torch::nn::Sequential(_make_layer(512, options.layers()[3], 2));

  fc = torch::nn::Linear(512 * options.expansion(), options.num_classes());
  register_module("conv1", conv1);
  register_module("bn1", bn1);
  register_module("layer1", layer1);
  register_module("layer2", layer2);
  register_module("layer3", layer3);
  register_module("layer4", layer4);
  register_module("fc", fc);
}

torch::Tensor ResNet101Impl::forward(torch::Tensor x) {
  x = conv1->forward(x);
  x = bn1->forward(x);
  x = torch::relu(x);
  x = torch::max_pool2d(x, 3, 2, 1);

  x = layer1->forward(x);
  x = layer2->forward(x);
  x = layer3->forward(x);
  x = layer4->forward(x);

  x = torch::avg_pool2d(x, 7, 1);
  x = x.view({x.sizes()[0], -1});
  x = fc->forward(x);

  // return torch::log_softmax(x, 1);
  return x;
}

std::vector<torch::Tensor> ResNet101Impl::features(torch::Tensor x) {
  std::vector<torch::Tensor> features;
  features.push_back(x);
  x = conv1->forward(x);
  x = bn1->forward(x);
  x = torch::relu(x);
  features.push_back(x);
  x = torch::max_pool2d(x, 3, 2, 1);

  x = layer1->forward(x);
  features.push_back(x);
  x = layer2->forward(x);
  features.push_back(x);
  x = layer3->forward(x);
  features.push_back(x);
  x = layer4->forward(x);
  features.push_back(x);

  return features;
}
