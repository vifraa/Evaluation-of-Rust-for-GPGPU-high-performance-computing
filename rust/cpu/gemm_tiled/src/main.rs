use cust::prelude::*;
use std::error::Error;

/// How many numbers to generate and add together.

static PTX: &str = include_str!("../../../resources/gemm_tiled.ptx");

fn main() -> Result<(), Box<dyn Error>> {
    let matrix_width: usize = env!("MATRIX_WIDTH").parse::<usize>()?;
    let block_width: usize = env!("BLOCK_WIDTH").parse::<usize>()?;

    let lhs = vec![1.0f32; matrix_width*matrix_width];
    let rhs = vec![2.0f32; matrix_width*matrix_width];

    let _ctx = cust::quick_init()?;

    // Make the CUDA module, modules just house the GPU code for the kernels we created.
    let module = Module::from_ptx(PTX, &[])?;


    // make a CUDA stream to issue calls to.
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // allocate the GPU memory needed to house our numbers and copy them over.
    let lhs_gpu = lhs.as_slice().as_dbuf()?;
    let rhs_gpu = rhs.as_slice().as_dbuf()?;


    // allocate our output buffer.
    let mut out = vec![0.0f32; matrix_width*matrix_width];
    let out_buf = out.as_slice().as_dbuf()?;

    // retrieve the matMulCUDA kernel from the module
    let func = module.get_function("matMulCUDA")?;


    let mut grid_width = matrix_width / block_width;
    if matrix_width % block_width != 0 {
        grid_width += 1;
    }

    unsafe {
        launch!(
            // slices are passed as two parameters, the pointer and the length.
            func<<<(grid_width as u32, grid_width as u32, 1), (block_width as u32, block_width as u32, 1), 0, stream>>>(
                lhs_gpu.as_device_ptr(),
                lhs_gpu.len(),
                rhs_gpu.as_device_ptr(),
                rhs_gpu.len(),
                out_buf.as_device_ptr(),
            )
        )?;
    }

    stream.synchronize()?;

    out_buf.copy_to(&mut out)?;

    println!("Finished.");

    Ok(())

}

