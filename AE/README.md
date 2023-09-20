# RECom Artifact Evaluation

The artifact contains the necessary software components to validate the main results in the RECom paper.
We provide a Dockfile for users to build the docker image, which contains the basic environment used to build and run the RECom examples.

Firstly, use the `Dockerfile` to build the docker image. Note: if the compute capability of your GPU is not 7.5 or 8.6, you should modify the `Dockerfile` correspondingly.

```bash
docker build -t recom:latest .
```

Then you can launch the container:

```bash
docker run -d --gpus all --net=host --name recom_ae -it recom:latest
docker exec -it recom_ae bash
```

After launching the container, clone this repo:

```bash
git clone --branch features/dev_pan --recurse-submodules https://github.com/PanZaifeng/recom.git recom
```

Then you can run a single script to perform the following steps:

1. Create the synthetic models E and F used in the paper.
2. Build the RECom addon.
3. Build the TensorFlow C++ examples.
4. Measure the inference latency of RECom and the TensorFlow baselines.
5. Draw the most important figures in the paper (Figures 10 and 11).

```bash
python recom/AE/build_and_run.py
```

Finally, you can open the generated figures to check the results.

Alternatively, you can run the `run_all.sh` script to perform all of the above steps:

```bash
./run_all.sh
```

