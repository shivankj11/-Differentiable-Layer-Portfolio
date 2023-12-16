# Differentiable-Layer-Portfolio
Fintech H2 Capstone

## Usage
Copy paste the following into your environment terminal (command line):
```
pip install -r requirements.txt
```

## Tasks / Ideas

#### Goal 1
- Portfolio indicator
- Portfolio strategy
- XAI
    - for features (compare to indicator)
    - for data
    - maybe OECR all the way back to data

#### Goal 2
- 1-step integrated approach
- Architecture change
- Loss function change
- XAI


## Full previous instruction

### Environment Setup
Ideally this is the command line flow for a linux ubuntu(WSL, github codespaces are also ubuntu style) based system setup. 
If you use debian system, maybe need to replace apt-get to yum. 
Please install conda before hand.  
``` shell
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
sh ./Miniconda3-py39_4.12.0-Linux-x86_64.sh
```
Env: github codespace (init state)

### Install
```
conda create -n qlib
source activate qlib 
conda install python=3.8
cd qlib
pip install numpy 
pip install --upgrade cython
pip install packaging
conda install hdf5 -y
pip install blosc2
pip install blosc
conda install -c anaconda lzo -y
sudo apt-get install libblosc-dev -y
conda install scikit-learn -y
conda install pytables -y
conda install matplotlib -y
conda install -c conda-forge statsmodels -y
conda install pybind11 -y
conda install mlflow -y
pip install pandas==1.5.3
pip install importlib-metadata==5.2.0
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y (if you use gpu: conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia  )
pip install cvxpylayers 
python setup.py install
python setup.py build_ext --inplace
```

### Daily Use
1. In command line: `source activate qlib`

2. Or directly run ipynb selecting "qlib" as your python kernel


### Troubleshooting
#### HTTPErr 409
``` shell
HTTPError: 409 Client Error: Public access is not permitted on this storage account. for url: https://qlibpublic.blob.core.windows.net/data/default/stock_data/v2/qlib_data_cn_1d_0.9.1.zip
```
Please enter experiment/ 
``` shell
cd experiement
source activate qlib
python initdatasetup.py

```

#### Executing Quick_Start.ipynb kernel die
Memory execeeded. Try to allocate more RAM codespaces or PC

#### Executing Quick_Start.ipynb occur Errors with word "home/xuecheng/"
Change EXP_NAME in the first cell to a different string (perhaps appending your name's abbr)
