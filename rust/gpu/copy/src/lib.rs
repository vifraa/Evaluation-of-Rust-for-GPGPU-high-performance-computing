#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]

use cuda_std::prelude::*;

extern crate alloc;
// Include ARRAY_SIZE here that is generated during compiling
// from environment variables.
include!("constants.rs");
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn copy(a: &[f32], c: *mut f32) {
    let thread_idx = (thread::block_dim().x * thread::block_idx().x + thread::thread_idx().x) as usize;
    if thread_idx < ARRAY_SIZE {
        let elem = &mut *c.add(thread_idx);
        //*elem = *a.get_unchecked(thread_idx);
        *elem = a[thread_idx];
    }
}

