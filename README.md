# libtorch_example
an example of libtorch

## Requirement
CUDA 10.1

cuDNN 8.0.5

GCC 7.3.0

libtorch Pre-cxx11 abi 1.7.1+cu101

## Directory Structure

```txt
.
├── 3rdParty
│   └── cuda-10.1
│   └── libtorch_1.7.1_cu101
├── src
│   └── example-app.cpp
├── CMakeLists.txt
└── README.md
```

## Build
```bash
cmake -B build
cmake --build build
```

## References
[get old version of libtorch](https://github.com/pytorch/pytorch/issues/40961)

[How To Run a pre-trained PyTorch model in C++](https://jumpml.com/howto-pytorch-c++/output/)

[libtorch（pytorch c++）教程（六）](https://zhuanlan.zhihu.com/p/369930932)

[LOADING A TORCHSCRIPT MODEL IN C++](https://pytorch.org/tutorials/advanced/cpp_export.html)