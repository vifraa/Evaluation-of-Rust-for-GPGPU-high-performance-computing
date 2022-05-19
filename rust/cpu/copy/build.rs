use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../../gpu/copy")
        .copy_to("../../resources/copy.ptx")
        .build()
        .unwrap();
}
