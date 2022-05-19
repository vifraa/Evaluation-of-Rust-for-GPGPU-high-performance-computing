use std::{fs::File, path::Path, io::Write};

fn main() {
    let out_dir = "./src/";
    let dest_path = Path::new(&out_dir).join("constants.rs");
    let mut f = File::create(&dest_path).expect("Could not create file");


    let matrix_width = option_env!("MATRIX_WIDTH").map_or(Ok(100), str::parse).expect("Could not parse MATRIX_WIDTH");

    writeln!(&mut f, "const MATRIX_WIDTH: usize = {};", matrix_width).unwrap();
    writeln!(&mut f, "const MATRIX_WIDTH_U32: u32 = {};", matrix_width).unwrap();
    
    println!("cargo:rerun-if-env-changed=MATRIX_WIDTH");
}
