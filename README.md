# K+ mass measurement

K+ mass measurement

The current code for the mass measurement uses full two sided Hypatia
distribution. This Hypatia pdf was coded in `lib99ocl` library to be able to
run in both CPU and GPU, in both OpenCL and CUDA.

By default the environment is set to use the first available device openCL
capable. This can be changed by setting
```
ipanema.initialize('opencl', 1)
```
to 
```
ipanema.initialize('cuda', 1)
```
which also will pick the first available device, but a CUDA capable.



## Installation
It is quite easy to install the package.

1. Download miniconda from [here](https://www.continuum.io/downloads). And 
install it.
2. Run the environment setup for conda:
```bash
conda env create -f environment.yml
conda activate kmass
```
3. Install all needed python packages:
```bash
python -m pip install -r requirements.txt
```


## Running the code
There is only one script to run, so it is easy:
```bash
pyton fitter_ipatia_extended.py
```
It takes 4 minutes in a M1 Macbook Air processor using CPU.



