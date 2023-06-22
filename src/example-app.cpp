#include <torch/script.h>
#include <torch/serialize.h>
#include <torch/torch.h>
#include <iostream>

class Detector {
 public:
  Detector(const std::string& model_path, bool use_gpu);
  ~Detector();

 private:
  torch::jit::script::Module model_;
  torch::Device device_;
  bool half_;
};

Detector::Detector(const std::string& model_path, bool use_gpu)
    : device_(torch::kCPU) {
  if (torch::cuda::is_available() && use_gpu) {
    std::cout << "use cuda..." << std::endl;
    device_ = torch::kCUDA;
  } else {
    std::cout << "use cpu..." << std::endl;
  }

  try {
    model_ = torch::jit::load(model_path);
    std::cout << "load success..." << std::endl;
  } catch (const c10::Error& e) {
    std::cerr << "Error loading the model" << std::endl;
    std::cerr << e.msg() << std::endl;
    std::exit(EXIT_FAILURE);
  }
  half_ = (device_ != torch::kCPU);
  model_.to(device_);

  if (half_) {
    model_.to(torch::kHalf);
  }
  model_.eval();
}

Detector::~Detector() {}

int main() {
  std::shared_ptr<Detector> detector =
      std::make_shared<Detector>("models/model.pt", true);
}