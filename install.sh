#!/bin/bash

### prerequisite
# sudo apt-get install freeglut3-dev

## create and activate custom conda env
# conda update --all
# conda create -n same python=3.8
# conda activate same

## torch 2.3 stable, CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install torch_geometric

pip install scipy tqdm IPython tensorboard matplotlib plotly pyyaml
pip install pynvml gputil

pip install glfw
pip install imgui
pip install opencv-python

python setup.py develop

conda install -y -c conda-forge -c matplotlib numpy scikit-learn scipy meshplot
pip install diffusers==0.21.4 einops==0.7.0 huggingface-hub==0.17.3 meshio==5.3.4 opencv-python==4.8.1.78 plyfile==1.0.1 transformers==4.34.1 trimesh==4.0.0 potpourri3d==1.0.0 robust_laplacian==0.2.7 accelerate==0.21.0
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# submodule: fairmotion (https://github.com/sunny-Codes/fairmotion)
git submodule init
git submodule update
cd src/fairmotion
python setup.py develop

