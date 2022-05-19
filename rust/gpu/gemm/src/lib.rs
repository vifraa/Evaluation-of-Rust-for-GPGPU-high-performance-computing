#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]


use cuda_std::prelude::*;

extern crate alloc;

// Include MATRIX_WIDTH and BLOCK_WIDTH here that is generated during compiling
// from environment variables.
include!("constants.rs");

const BLOCK_WIDTH: u32 = 16;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn matMulCUDA(a: &[f32], b: &[f32], c: *mut f32) {
    let thread_idx = thread::thread_idx();
    let block_idx = thread::block_idx();

    let row = (block_idx.y * BLOCK_WIDTH + thread_idx.y) as usize;
    let column = (block_idx.x * BLOCK_WIDTH + thread_idx.x) as usize;

    if row < MATRIX_WIDTH && column < MATRIX_WIDTH {
        let mut p_value: f32 = 0.0;
        for k in 0..MATRIX_WIDTH {
            //p_value += a.get_unchecked(row*MATRIX_WIDTH+k)*b.get_unchecked(k*MATRIX_WIDTH+column);
            p_value += a[row*MATRIX_WIDTH+k]*b[k*MATRIX_WIDTH+column];
        }

        let write = &mut *c.add(row*MATRIX_WIDTH+column);
        *write = p_value;
    }
}
