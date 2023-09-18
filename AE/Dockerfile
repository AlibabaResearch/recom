FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu18.04

RUN apt-get update

# install python3.8 and pip
RUN apt-get install -y curl python3.8 python3.8-distutils
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

RUN pip install tensorflow==2.6.2
RUN pip install protobuf==3.20

# prepare the codes
RUN apt-get install -y git
RUN git clone --branch features/dev_pan --recurse-submodules https://github.com/PanZaifeng/recom.git recom

RUN curl https://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/build_deps/bazelisk/v1.18.0/bazelisk-linux-amd64 -o /usr/local/bin/bazel
RUN chmod +x /usr/local/bin/bazel

# configure TF addon
WORKDIR recom
ENV TF_NEED_CUDA=1 \
    TF_CUDA_VERSION=11 \
    TF_CUDNN_VERSION=8 \
    CUDA_TOOLKIT_PATH=/usr/local/cuda \
    CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu
RUN python configure.py

RUN apt-get install -y libgmp-dev
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# build TF addon
# RUN bazel build //tensorflow_addons:librecom.so
# RUN bazel build //tensorflow_addons:libtf_cpu_gpu.so

RUN bash examples/cc/apply_patch.sh

# configure TF
ENV TF_CONFIGURE_IOS=False \
    TF_SET_ANDROID_WORKSPACE=False \
    TF_NEED_ROCM=0 \
    TF_CUDA_CLANG=0 \
    TF_NEED_TENSORRT=0 \
    GCC_HOST_COMPILER_PATH=/usr/bin/gcc \
    CC_OPT_FLAGS=-Wno-sign-compare \
    PYTHON_BIN_PATH=/usr/bin/python \
    USE_DEFAULT_PYTHON_LIB_PATH=1
ENV TF_CUDA_COMPUTE_CAPABILITIES=7.5,8.6
WORKDIR examples/cc/tensorflow-v2.6.2
RUN ./configure

# build TF examples
# RUN bazel build --config=cuda --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/recom:benchmark_multi_thread
# RUN bazel build --config=cuda --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/recom:benchmark_throughput

WORKDIR ../../..
RUN pip install matplotlib==3.3.4
