#include "resnet101_impl.h"
#include <ATen/Functions.h>
#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/serialize.h>
#include <torch/torch.h>
#include "block.h"
#include "conv_options.h"

torch::nn::Sequential ResNet101Impl::_make_layer(int64_t planes, int64_t blocks,
                                                 int64_t stride) {
  torch::nn::Sequential downsample;
  if (stride != 1 || inplanes != planes * expansion) {
    downsample = torch::nn::Sequential(
        torch::nn::Conv2d(
            conv_options(inplanes, planes * expansion, 1, stride, 0, 1, false)),
        torch::nn::BatchNorm2d(planes * expansion));
  }
  torch::nn::Sequential layers;
  layers->push_back(Block(inplanes, planes, stride, downsample, groups,
                          base_width, is_basic));
  inplanes = planes * expansion;
  for (int64_t i = 1; i < blocks; i++) {
    layers->push_back(Block(inplanes, planes, 1, torch::nn::Sequential(),
                            groups, base_width, is_basic));
  }

  return layers;
}

ResNet101Impl::ResNet101Impl(std::vector<int> layers, int num_classes,
                             std::string model_type, int _groups,
                             int _width_per_group) {
  if (model_type != "resnet18" && model_type != "resnet34") {
    expansion = 4;
    is_basic = false;
  }
  groups = _groups;
  base_width = _width_per_group;
  conv1 = torch::nn::Conv2d(conv_options(3, 64, 7, 2, 3, 1, false));
  bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
  layer1 = torch::nn::Sequential(_make_layer(64, layers[0]));
  layer2 = torch::nn::Sequential(_make_layer(128, layers[1], 2));
  layer3 = torch::nn::Sequential(_make_layer(256, layers[2], 2));
  layer4 = torch::nn::Sequential(_make_layer(512, layers[3], 2));

  fc = torch::nn::Linear(512 * expansion, num_classes);
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

  return torch::log_softmax(x, 1);
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
