# RECom Python Examples

This directory contains examples of RECom by using the TensorFlow Python interface. You can use these scripts to create the synthetic models evaluated in our paper. Please make sure TensorFlow 2.6 is installed before running.

To create the model E and F in the paper, run `dlrm.py` directly.
Then, you will find the saved models named `DLRM1` and `DLRM2` in the current path.

```bash
python dlrm.py
```

To create the microbenchmark models in the paper, run `microbenchmark.py` with a specified number of embedding columns and exported model path.

```bash
python microbenchmark.py -N 100 --save_path microbenchmark_100
```

**In our evaluation, we use TensorFlow's C++ interface rather than its Python interface.**
The reason behind this decision is that Python can introduce a significant overhead when processing model inputs, especially when the number of inputs becomes very large (e.g., over 1000).
Additionally, due to the Global Interpreter Lock (GIL), it is not possible to run multiple serving threads in parallel. Duplicating instances in different processes can also result in an extremely large memory overhead that may exceed the GPU's capacity.

However, if you prefer not to build the C++ examples and want to test the effectiveness of RECom quickly, you can still run the scripts as follows:

```bash
python microbenchmark.py -N 100 --lib_path /path/to/recom/bazel-bin/tensorflow_addons/librecom.so
```
