# Evaluation of Rust for GPGPU high performance computing 
***A performance comparison between native C++-CUDA kernels and Rust-CUDA kernels on NVIDIA hardware***  

Authors: *Carl Östling* and *Viktor Franzén*


### Content 
This repository contains the source code referenced in the thesis "Evaluation of Rust for GPGPU high performance computing".

Each subfolder contains the implementation for each respective language and kernel.

**gemm**       -> Matrix Multiplication without tiling  
**gemm_tiled** -> Matrix Multiplication with tiling  
**copy**       -> Array copy
