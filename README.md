# libtorch_example
an example of libtorch

## Requirement
CUDA 10.1

cuDNN 8.0.5

GCC 7.3.0

libtorch Pre-cxx11 abi 1.7.1+cu101

## Directory Structure

```
.
├── 3rdParty
│   ├── cuda-10.1
│   └── libtorch_1.7.1_cu101
├── include
│   ├── block.h
│   ├── conv_options.h
│   └── resnet101_impl.h
├── src
│   ├── block.cpp
│   ├── example-app.cpp
│   ├── load-model-jit.cpp
│   ├── load-model.cpp
│   └── resnet101_impl.cpp
├── CMakeLists.txt
└── README.md
```

ResNet101的实现：
    block.h block.cpp resnet101_impl.h resnet101_impl.cpp conv_options.h

example-add.cpp：加载TorchScript模型到GPU，并eval。

load-model-jit.cpp：同上。

load-model.cpp：使用libtorch定义模型结构，然后使用`torch::load`加载模型。

## Build
```bash
cmake -B build
cmake --build build
```

## References
[How To Run a pre-trained PyTorch model in C++](https://jumpml.com/howto-pytorch-c++/output/)

[libtorch（pytorch c++）教程（六）](https://zhuanlan.zhihu.com/p/369930932)

[LOADING A TORCHSCRIPT MODEL IN C++](https://pytorch.org/tutorials/advanced/cpp_export.html)