#!/bin/bash
dir=$(dirname "$0")
cd $dir/tensorflow-v2.6.2
git apply ../recom_examples.patch
