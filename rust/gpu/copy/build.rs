use std::{fs::File, path::Path, io::Write};

fn main() {
    let out_dir = "./src/";
    let dest_path = Path::new(&out_dir).join("constants.rs");
    let mut f = File::create(&dest_path).expect("Could not create file");


    let array_size = option_env!("ARRAY_SIZE").map_or(Ok(100), str::parse).expect("Could not parse ARRAY_SIZE");

    writeln!(&mut f, "const ARRAY_SIZE: usize = {};", array_size).unwrap();

    println!("cargo:rerun-if-env-changed=MATRIX_WIDTH");
    println!("cargo:rerun-if-env-changed=BLOCK_WIDTH");
}
