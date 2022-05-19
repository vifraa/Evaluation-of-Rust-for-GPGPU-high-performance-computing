use std::{fs::File, path::Path, io::Write};

fn main() {
    let out_dir = "./src/";
    let dest_path = Path::new(&out_dir).join("constants.rs");
    let mut f = File::create(&dest_path).expect("Could not create file");


    let matrix_width = option_env!("MATRIX_WIDTH").map_or(Ok(100), str::parse).expect("Could not parse MATRIX_WIDTH");
    let block_width = option_env!("BLOCK_WIDTH").map_or(Ok(32), str::parse).expect("Could not parse BLOCK_WIDTH");

    writeln!(&mut f, "const MATRIX_WIDTH: usize = {};", matrix_width).unwrap();
    writeln!(&mut f, "const MATRIX_WIDTH_U32: u32 = {};", matrix_width).unwrap();
    writeln!(&mut f, "const BLOCK_WIDTH: usize = {};", block_width).unwrap();
    writeln!(&mut f, "const BLOCK_WIDTH_U32: u32 = {};", block_width).unwrap();

    println!("cargo:rerun-if-env-changed=MATRIX_WIDTH");
    println!("cargo:rerun-if-env-changed=BLOCK_WIDTH");
}
