# Copyright 2023 The RECom Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse


def os_check(command):
    status = os.system(command)
    if status != 0:
        raise Exception(f"Process exit status {status}, command: {command}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
    parser.add_argument("--tf_cuda_cc", type=str, default="7.5,8.6")
    args = parser.parse_args()

    if args.tf_cuda_cc:
        os.environment["TF_CUDA_COMPUTE_CAPABILITIES"] = args.tf_cuda_cc

    ae_dir = os.path.dirname(os.path.abspath(__file__))
    recom_dir = f"{ae_dir}/.."

    log_dir = f"{ae_dir}/logs"
    os.makedirs(log_dir, exist_ok=True)

    # create models
    model_dir = f"{recom_dir}/models"
    os.makedirs(model_dir, exist_ok=True)
    os.chdir(model_dir)
    os_check(f"python {recom_dir}/examples/python/dlrm.py")

    # configure and build RECom addon
    os.chdir(recom_dir)
    os_check("python configure.py")
    os_check("bazel build //tensorflow_addons:librecom.so")
    os_check("bazel build //tensorflow_addons:libtf_cpu_gpu.so")

    addon_dir = f"{recom_dir}/bazel-bin/tensorflow_addons"
    librecom_path = f"{addon_dir}/librecom.so"
    libtf_cpu_gpu_path = f"{addon_dir}/libtf_cpu_gpu.so"

    # configure and build TF cc examples
    tf_dir = f"{recom_dir}/examples/cc/tensorflow-v2.6.2"
    os.chdir(tf_dir)
    os_check("../apply_patch.sh")
    os_check("./configure")
    os_check('bazel build --config=cuda --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/recom:benchmark_multi_thread')
    os_check('bazel build --config=cuda --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/recom:benchmark_throughput')

    cc_bin_dir = f"{tf_dir}/bazel-bin/tensorflow/recom"
    latency_script = f"{cc_bin_dir}/benchmark_multi_thread"
    throughput_script = f"{cc_bin_dir}/benchmark_throughput"

    pure_cpu_set = f"CUDA_VISIBLE_DEVICES=-1 taskset -c 0-31"
    cpu_gpu_set = f"CUDA_VISIBLE_DEVICES=0 taskset -c 0-3"

    # measure latency
    for m in ["E", "F"]:
        for bs in [32, 64, 128, 256, 512, 1024, 2048]:
            main_body = f"{latency_script} --batch_size {bs} --model_path {model_dir}/{m}"
            # TF-CPU
            os_check(f"{pure_cpu_set} {main_body} --disable_gpu 2>&1 | tee {log_dir}/l_tf_cpu_{m}_{bs}.log")
            # TF-GPU
            os_check(f"{cpu_gpu_set} {main_body} 2>&1 | tee {log_dir}/l_tf_gpu_{m}_{bs}.log")
            # TF-CPU-GPU
            os_check(f"{cpu_gpu_set} {main_body} --lib_path {libtf_cpu_gpu_path} 2>&1 | tee {log_dir}/l_tf_cpu_gpu_{m}_{bs}.log")
            # RECom
            os_check(f"{cpu_gpu_set} {main_body} --lib_path {librecom_path} 2>&1 | tee {log_dir}/l_recom_{m}_{bs}.log")

    # measure throughput
    for m in ["E", "F"]:
        for n in [2, 4, 8]:
            main_body = f"{throughput_script} --serve_workers {n} --model_path {model_dir}/{m}"
            # TF-CPU
            os_check(f"{pure_cpu_set} {main_body} --disable_gpu 2>&1 | tee {log_dir}/t_tf_cpu_{m}_{n}.log")
            # RECom
            os_check(f"{cpu_gpu_set} {main_body} --lib_path {librecom_path} 2>&1 | tee {log_dir}/t_recom_{m}_{n}.log")

    # plot figures
    os.chdir(ae_dir)
    os_check(f"python plot_latency.py --log_dir {log_dir} --output {ae_dir}/latency.pdf")
    os_check(f"python plot_throughput.py --log_dir {log_dir} --output {ae_dir}/throughput.pdf")
