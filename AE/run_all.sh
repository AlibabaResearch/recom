#!/bin/bash
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

docker build -f ./Dockerfile -t recom:latest .
docker run -d --gpus all --net=host --name recom_ae -it recom:latest
docker exec recom_ae bash -c "git clone --recurse-submodules https://github.com/AlibabaResearch/recom.git recom && python recom/AE/build_and_run.py"
docker cp recom_ae:recom/AE/latency.pdf .
docker cp recom_ae:recom/AE/throughput.pdf .
