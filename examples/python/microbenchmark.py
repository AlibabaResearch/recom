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

import numpy as np
import argparse
import random
import time
import os
import sys
import shutil

try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf

tf.disable_resource_variables()
tf.disable_eager_execution()
tf.random.set_random_seed(0)
np.random.seed(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--num_columns', type=int, required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('--embedding_table_rows', type=int, default=100)
    parser.add_argument('--embedding_dim', type=int, default=8)
    parser.add_argument('--lib_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--random_boundary', action='store_true')
    args = parser.parse_args()

    N = args.num_columns
    bs = args.batch_size
    nrows = args.embedding_table_rows
    dim = args.embedding_dim
    boundaries = list(range(0, nrows * 5, 5))

    columns = []
    features = {}
    feed_dict = {}
    input_sig = {}
    for i in range(N):
        feat_name = f'f{i}'
        placeholder = tf.placeholder(dtype=tf.float32, name=feat_name, shape=[None])
        nc = tf.feature_column.numeric_column(feat_name)
        if args.random_boundary:
            step = random.randint(5, 10)
            nrows = args.embedding_table_rows + random.randint(-50, 50)
            boundaries = list(range(0, nrows * step, step))
        bc = tf.feature_column.bucketized_column(nc, boundaries=boundaries)
        ec = tf.feature_column.embedding_column(bc, 8, combiner='mean')

        features[feat_name] = placeholder
        columns.append(ec)

        feed_dict[f'{feat_name}:0'] = np.random.randint(-1, 10000, size=[bs])
        input_sig[feat_name] = placeholder

    output = tf.feature_column.input_layer(features, columns)

    if args.lib_path:
        tf.load_op_library(args.lib_path)
    config = tf.ConfigProto(intra_op_parallelism_threads=0,
                            inter_op_parallelism_threads=0,
                            device_count={'GPU': 0})
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(output, feed_dict)
        if args.save_path:
            shutil.rmtree(args.save_path, ignore_errors=True)
            tf.saved_model.simple_save(
                sess, args.save_path, input_sig, {'output': output})

        for _ in range(10):
            sess.run(output, feed_dict)

        t1 = time.time()
        for _ in range(100):
            sess.run(output, feed_dict)
        t2 = time.time()
        print(f'{(t2 - t1) / 100 * 1e3}ms')

