# RECom

RECom is an ML compiler that aims to accelerate the expensive embedding column processing during the inference of deep recommendation models. It fuses massive embedding columns into a single kernel and maps each column into a separate threadblock on the GPU.

Currently, RECom is implemented as a TensorFlow add-on based on [TensorFlow Addons](https://github.com/tensorflow/addons) using C++.
We also utilize the [SymEngine Library](https://github.com/symengine/symengine) to perform symbolic expression computations to handle dynamic shapes.

## Build RECom

We highly recommend building RECom in the docker `nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04`, as the compatibility of RECom has not been checked yet.

You should prepare the following environments before building RECom:

* Python 3.8

* TensorFlow 2.6.2

* Bazel 4.2.1 ([bazelisk](https://github.com/bazelbuild/bazelisk) is recommended)

You should also download the libgmp-dev, which is required by SymEngine:

```bash
apt-get install libgmp-dev
```

After preparing the environments, you should set some environment variables and run the `configure.py` to generate the `.bazelrc`:

```bash
export TF_NEED_CUDA="1"

# Set these if the below defaults are different on your system
export TF_CUDA_VERSION="11"
export TF_CUDNN_VERSION="8"
export CUDA_TOOLKIT_PATH="/usr/local/cuda"
export CUDNN_INSTALL_PATH="/usr/lib/x86_64-linux-gnu"

python ./configure.py
```

Then, you can start building the shared library of RECom:

```bash
bazel build //tensorflow_addons:librecom.so
```

Finally, you will find the target `librecom.so` in `bazel-bin/tensorflow_addons`.

## Usage

You can use `tf.load_op_library`/`TF_LoadLibrary` in your Python/C++ inference scripts to load the TensorFlow addon of RECom without modifying any source codes of the models.
Then, the models will be optimized automatically (warm-up required).

More details can be found in the `examples` folder.

## Publication

**[ASPLOS'24] RECom: A Compiler Approach to Accelerating Recommendation Model Inference with Massive Embedding Columns**

Zaifeng Pan, Zhen Zheng, Feng Zhang, Ruofan Wu, Hao Liang, Dalin Wang, Xiafei Qiu, Junjie Bai, Wei Lin, Xiaoyong Du
