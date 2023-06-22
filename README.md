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