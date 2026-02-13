# Luna: Bound Propagator for Neural Network Analysis

## Introduction

Luna is a bound propagation egine for neural network verification engine written in C++17with Python bindings. It implements state-of-the-art linear relaxation basedperturbation analysis (LiRPA) algorithms, including
[CROWN](https://arxiv.org/pdf/1811.00866.pdf) and
[Alpha-CROWN](https://arxiv.org/pdf/2011.13824.pdf), to compute guaranteed
output bounds for neural networks under input perturbations.

Luna accepts models in [ONNX](https://onnx.ai/) format and property
specifications in [VNN-LIB](https://www.vnnlib.org/) format, making it
compatible with standard neural network verification benchmarks such as
[VNN-COMP](https://sites.google.com/view/vnn2025).

Our library supports the following bound propagation algorithms:

* Interval Bound Propagation ([IBP](https://arxiv.org/pdf/1810.12715.pdf))
* Backward mode LiRPA bound propagation ([CROWN](https://arxiv.org/pdf/1811.00866.pdf)/[DeepPoly](https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf))
* Backward mode LiRPA with optimized bounds ([Alpha-CROWN](https://arxiv.org/pdf/2011.13824.pdf))

Luna supports a wide range of neural network layer types, including fully
connected, convolutional, transposed convolutional, batch normalization, ReLU,
sigmoid, reshape, flatten, slice, concatenation, and residual (add/sub)
connections.

## Technical Background

Neural network verification asks the question: given a neural network and a
bounded region of inputs, what are the guaranteed bounds on the network's
outputs?

**IBP** (Interval Bound Propagation) computes fast but loose bounds by
propagating intervals forward through each layer.

**CROWN** computes tighter bounds by propagating linear relaxations *backward*
through the computational graph, accumulating symbolic bound expressions that
are concretized at the input layer:

```
output >= A * input + b    (lower bound)
output <= A * input + b    (upper bound)
```

**Alpha-CROWN** further tightens CROWN bounds by introducing learnable slope
parameters (alpha) at nonlinear nodes (e.g., ReLU). These parameters are
optimized via gradient descent to minimize the bound gap:

```
for each iteration:
    run CROWN with current alphas
    compute loss = sum of bound widths
    update alphas via Adam optimizer
```

## Architecture

```
luna/
├── src/
│   ├── engine/
│   │   ├── TorchModel            # Main model class and computational graph
│   │   ├── CROWNAnalysis         # CROWN backward bound propagation
│   │   ├── AlphaCROWNAnalysis    # Alpha-CROWN optimization loop
│   │   └── nodes/                # Per-layer bound implementations
│   │       ├── BoundedLinearNode
│   │       ├── BoundedConvNode
│   │       ├── BoundedReLUNode   # Alpha-optimizable
│   │       ├── BoundedSigmoidNode
│   │       └── ...               # 15+ node types
│   ├── input_parsers/
│   │   ├── OnnxToTorch           # ONNX model parser
│   │   └── VnnLibInputParser     # VNN-LIB property parser
│   └── configuration/
│       └── LunaConfiguration     # Static configuration settings
├── lunapy/                       # Python bindings (pybind11)
│   ├── lunapy.py                 # High-level Python API
│   └── examples/                 # Tutorial examples
├── auto_LiRPA/                   # Modified auto_LiRPA (reference/comparison)
├── tests/
│   ├── unit/                     # CxxTest unit tests
│   ├── integration/              # End-to-end verification tests
│   └── property/                 # Invariant-based tests
└── resources/
    ├── onnx/                     # ONNX model files
    └── properties/               # VNN-LIB specifications
```

## Installation

### Requirements

- CMake 3.16+
- C++17 compiler
- PyTorch 2.2.1+ (libtorch)

The following dependencies are downloaded automatically by the build system if
not found:

- Boost 1.84.0
- Protobuf 3.19.2
- ONNX 1.15.0
- pybind11 2.11.1

### Building

```bash
git clone <repository-url>
cd luna
mkdir build && cd build
cmake ../
make -j$(nproc)
```

## Quick Start

### Command Line

Run verification on an ONNX model with a VNN-LIB property specification:

```bash
luna --input model.onnx --vnnlib property.vnnlib
```

For example, verifying an ACAS Xu collision avoidance network:

```bash
luna --input resources/regular_benchmarks/benchmarks/acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx --property resources/regular_benchmarks/benchmarks/acasxu_2023/vnnlib/prop_3.vnnlib --method alpha-crown --optimize-lower --optimize-upper --lr 0.5 --iterations 20

```
