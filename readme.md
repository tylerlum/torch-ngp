# Torch Neural Graphics Primatives - MSL Fork

This is a fork of the repository [torch-ngp](https://github.com/ashawkey/torch-ngp), which is itself based on [instant-ngp](https://github.com/NVlabs/instant-ngp) by Thomas Müller.

The goal of this repository is to provide an fast and easy to use implementation of basic NeRF utilities for more efficient research iteration.

* MSL verified timing results:
    - LEGO RESULTS
    - FOX RESULTS

## Installation
1) Clone this repository
    ```bash
    git clone --recursive git@github.com:StanfordMSL/torch-ngp.git
    ```
    * Make sure to use the recursive argument because of the `cutlass` submodule which will otherwise throw errors for some functionality.
    * If you want to add this repo as a submodule to a current project then use:
        ```bash
        git submodule add git@github.com:StanfordMSL/torch-ngp.git
        cd torch-ngp
        git submodule update --init --recursive # To add cutlass
        ```

2) Setup a Python environment 
    ```bash
    cd torch-ngp
    virtualenv venv # instructions for virtualenv but should be similar with conda etc.
    source venv/bin/activate
    pip install -r requirements.txt
    pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
    
    # To install the torch_ngp package
    pip install -e .
    ```

3) Download the basic NeRF Datasets
    ```bash
    cd ... # wherever you want your data
    mkdir -p data
    cd data
    wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
    unzip nerf_example_data.zip
    cd ..
    ```

## Usage
There are a variety of ways to use this repository. The `main_nerf.py` script uses all of the functionality of the original repository to run NeRF examples. Find the usage instructions for this script in the documentation of [torch-ngp](https://github.com/ashawkey/torch-ngp).

Alternatively, `nerf_basic.py` shows a more stripped back implementation of the core functionality of the packages in this repository.

## Goals
    - [] Pose optimization functionality
    - [] Benchmark results
