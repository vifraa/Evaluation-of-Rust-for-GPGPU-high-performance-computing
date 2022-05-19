#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]


use cuda_std::prelude::*;
use cuda_std::shared_array;

extern crate alloc;

// Include MATRIX_WIDTH and BLOCK_WIDTH here that is generated during compiling
// from environment variables.
include!("constants.rs");

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn matMulCUDA(a: &[f32], b: &[f32], c: *mut f32) {

    let a_shared_pointer = shared_array![f32; BLOCK_WIDTH*BLOCK_WIDTH];
    let b_shared_pointer = shared_array![f32; BLOCK_WIDTH*BLOCK_WIDTH];

    let thread_idx = thread::thread_idx();
    let block_idx = thread::block_idx();
    
    let row_idx = block_idx.y * BLOCK_WIDTH_U32 + thread_idx.y;
    let col_idx = block_idx.x * BLOCK_WIDTH_U32 + thread_idx.x;


    let mut computed_value: f32 = 0.0;
    
    for m in 0..(MATRIX_WIDTH_U32 +BLOCK_WIDTH_U32-1)/BLOCK_WIDTH_U32{
        //copy to shared memory
        if m * BLOCK_WIDTH_U32 + thread_idx.x < MATRIX_WIDTH_U32 && row_idx < MATRIX_WIDTH_U32{
            let mut a_write = &mut *a_shared_pointer.add((thread_idx.y*BLOCK_WIDTH_U32 + thread_idx.x) as usize);
            //*a_write = *a.get_unchecked(( row_idx*MATRIX_WIDTH_U32 + m*BLOCK_WIDTH_U32+thread_idx.x ) as usize);
            *a_write = a[( row_idx*MATRIX_WIDTH_U32 + m*BLOCK_WIDTH_U32+thread_idx.x ) as usize];
        }
        else{

            let mut a_write = &mut *a_shared_pointer.add((thread_idx.y*BLOCK_WIDTH_U32 + thread_idx.x) as usize);
            *a_write = 0.0;

        }

        if m*BLOCK_WIDTH_U32 + thread_idx.y < MATRIX_WIDTH_U32 && col_idx < MATRIX_WIDTH_U32 {
            let mut b_write = &mut *b_shared_pointer.add((thread_idx.y*BLOCK_WIDTH_U32 + thread_idx.x) as usize);
            //*b_write = *b.get_unchecked(((m*BLOCK_WIDTH_U32 + thread_idx.y)*MATRIX_WIDTH_U32+col_idx) as usize);
            *b_write = b[((m*BLOCK_WIDTH_U32 + thread_idx.y)*MATRIX_WIDTH_U32+col_idx) as usize];
        }
        else{
            let mut b_write = &mut *b_shared_pointer.add((thread_idx.y*BLOCK_WIDTH_U32 + thread_idx.x) as usize);
            *b_write = 0.0;
        }

        //sync to make sure all data is available in shared memory before computations
        thread::sync_threads();

        for k in 0..(BLOCK_WIDTH_U32){
            let mut a_read = &mut *a_shared_pointer.add((thread_idx.y*BLOCK_WIDTH_U32 + k) as usize);
            let mut b_read = &mut *b_shared_pointer.add((k*BLOCK_WIDTH_U32 + thread_idx.x) as usize);
            computed_value += *a_read * *b_read;
        }

        //sync to ensure all threads finished using shared memory before we move
        thread::sync_threads();
    }


    if row_idx < MATRIX_WIDTH_U32 && col_idx < MATRIX_WIDTH_U32{

        let elem = &mut *c.add((row_idx * MATRIX_WIDTH_U32 + col_idx) as usize);
        *elem = computed_value;
    }
}
