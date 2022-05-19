use cust::prelude::*;
use std::error::Error;


static PTX: &str = include_str!("../../../resources/copy.ptx");

fn main() -> Result<(), Box<dyn Error>> {
    let array_size: usize = env!("ARRAY_SIZE").parse::<usize>()?;
    let a = vec![1.0f32; array_size];

    let _ctx = cust::quick_init()?;

    // Make the CUDA module, modules just house the GPU code for the kernels we created.
    let module = Module::from_ptx(PTX, &[])?;

    // make a CUDA stream to issue calls to.
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // allocate the GPU memory
    let a_gpu = a.as_slice().as_dbuf()?;

    // allocate our output buffer.
    let mut out = vec![0.0f32; array_size];
    let out_buf = out.as_slice().as_dbuf()?;

    // retrieve the copy kernel from the module
    let func = module.get_function("copy")?;

    let thr_per_blk = 512;
    let grid_size = (array_size as u32 + thr_per_blk - 1) / thr_per_blk;

    unsafe {
        launch!(
            // slices are passed as two parameters, the pointer and the length.
            func<<<(grid_size,1,1), (thr_per_blk, 1, 1), 0, stream>>>(
                a_gpu.as_device_ptr(),
                a_gpu.len(),
                out_buf.as_device_ptr(),
            )
        )?;
    }

    stream.synchronize()?;
    out_buf.copy_to(&mut out)?;

    for i in 0..array_size {
        let tolerance = 0.0000001;
        if (a[i] - out[i]).abs() > tolerance {
            println!("\nError: value of C[{}] = {} instead of A[i]={}\n", i, out[i],a[i]);

            panic!();
        }
    }

    print!("\n---------------------------\n");
    print!("__SUCCESS__\n");
    print!("---------------------------\n");
    print!("N                 = {}\n", array_size);
    print!("Threads Per Block = {}\n", thr_per_blk);
    print!("Blocks In Grid    = {}\n", grid_size);
    print!("---------------------------\n\n");



    Ok(())
}
