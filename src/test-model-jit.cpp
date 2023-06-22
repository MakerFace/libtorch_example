#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>
#include <iostream>

//* https://pytorch.org/tutorials/advanced/cpp_export.html

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage : example \n";
    return -1;
  }

  torch::jit::script::Module model;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load(argv[1]);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    std::cerr << e.msg() << std::endl;
    return -1;
  }
  std::cout << "ok\n";

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1, 3, 224, 224}));

  // Execute the model and turn its output into a tensor.
  at::Tensor output = model.forward(inputs).toTensor();

  std::cout << output.slice(1, 0, 5) << '\n';
}