#ifndef CONV_OPTIONS_H
#define CONV_OPTIONS_H

#include <torch/torch.h>

inline torch::nn::Conv2dOptions conv_options(int64_t in_planes,
                                             int64_t out_planes,
                                             int64_t kerner_size,
                                             int64_t stride, int64_t padding,
                                             int groups, bool with_bias) {
  torch::nn::Conv2dOptions conv_options =
      torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
  conv_options.stride(stride);
  conv_options.padding(padding);
  conv_options.bias(with_bias);
  conv_options.groups(groups);
  return conv_options;
}

#endif  // CONV_OPTIONS_H