# RECom C++ Examples

This directory contains examples of RECom using TensorFlow C++. You can use these scripts to reproduce the evaluation results in our paper.

You can follow the subsequent instructions to run the C++ examples. Please do **NOT** build them in this repository directly.

## How to build

To run TensorFlow C++ examples, you should first clone and checkout the TensorFlow source codes:

```bash
git clone --branch v2.6.2 https://github.com/tensorflow/tensorflow.git
```

Second, install the necessary Python packages:

```bash
pip install numpy==1.19
pip install keras_preprocessing --no-deps
```

Then, you can run the `configure` script in the TensorFlow repository to launch a configuration session.
You should enter 'y' for the CUDA support only and specify the Python and CUDA paths according to your systems.

```bash
cd tensorflow
./configure
```

After configuration, copy the C++ examples (this folder) to `tensorflow/recom` under the TensorFlow directory and rename `BUILD_` to `BUILD`:

```bash
cp /path/to/recom/examples/c++ -r tensorflow/recom
mv tensorflow/recom/BUILD_ tensorflow/recom/BUILD
```

Finally, build the targets:

```bash
bazel build --config=cuda --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/recom:benchmark_multi_thread
```

More details about how to run TensorFlow C++ examples can be found in [TensorFlow - Build from source](https://www.tensorflow.org/install/source).


## Run the examples

After building, you can find the executable binaries in `bazel-bin/tensorflow/recom` under the TensorFlow directory.
Use `-h` or `--help` to show the usage (we use [cxxopts](https://github.com/jarro2783/cxxopts) to parse the arguments).

You can run the Python examples in this repository to create TensorFlow saved models:

```bash
python examples/python/microbenchmark.py -N 1000 --save_path microbenchmark_1000
```

Then, execute the binaries of C++ examples:

```bash
# TF-GPU
./bazel-bin/tensorflow/recom/benchmark_multi_thread --model_path /path/to/recom/microbenchmark_1000 --batch_size 256 --serve_workers 1

# TF-CPU
./bazel-bin/tensorflow/recom/benchmark_multi_thread --model_path /path/to/recom/microbenchmark_1000 --batch_size 256 --serve_workers 1 --disable_gpu

# RECom
./bazel-bin/tensorflow/recom/benchmark_multi_thread --model_path /path/to/recom/microbenchmark_1000 --batch_size 256 --serve_workers 1 --lib_path /path/to/recom/bazel-bin/tensorflow_addons/librecom.so
```
