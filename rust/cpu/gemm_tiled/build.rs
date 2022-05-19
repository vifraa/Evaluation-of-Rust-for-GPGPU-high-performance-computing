use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../../gpu/gemm_tiled")
        .copy_to("../../resources/gemm_tiled.ptx")
        .build()
        .unwrap();
}
