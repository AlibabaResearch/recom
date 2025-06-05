FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu18.04

RUN apt-get update

# install python3.8 and pip
RUN apt-get install -y curl python3.8 python3.8-distutils
RUN curl https://bootstrap.pypa.io/pip/3.8/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

RUN pip install tensorflow==2.6.2
RUN pip install protobuf==3.20

RUN apt-get install -y git 
RUN apt-get install -y libgmp-dev

RUN curl https://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/build_deps/bazelisk/v1.18.0/bazelisk-linux-amd64 -o /usr/local/bin/bazel
RUN chmod +x /usr/local/bin/bazel

ENV TF_NEED_CUDA=1 \
    TF_CUDA_VERSION=11 \
    TF_CUDNN_VERSION=8 \
    CUDA_TOOLKIT_PATH=/usr/local/cuda \
    CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

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

RUN pip install matplotlib==3.3.4
