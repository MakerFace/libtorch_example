
#include <torch/script.h>
#include <torch/serialize.h>
#include <torch/torch.h>
#include "resnet101_impl.h"

/**
 * failure [enforce fail at inline_container.cc:222] . file not found: archive/constants.pkl
 */

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage : example \n";
    return -1;
  }

  ResNet101 model;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    std::cout << "use torch::load loading model" << std::endl;
    torch::load(argv[1]);
    torch::load(model, argv[1]);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    std::cerr << e.msg() << std::endl;
    return -1;
  }
  std::cout << "ok\n";

  // Create a vector of inputs.
  torch::Tensor inputs = torch::ones({1, 3, 224, 224});

  // Execute the model and turn its output into a tensor.
  at::Tensor output = model->forward(inputs);

  std::cout << output.slice(1, 0, 5) << '\n';
}