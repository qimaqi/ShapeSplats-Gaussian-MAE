#!/bin/bash
conda create -n gaussian_mae python=3.9 -y 
conda activate gaussian_mae

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt