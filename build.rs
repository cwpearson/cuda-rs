extern crate bindgen;
extern crate find;
extern crate glob;
extern crate nvcc;

use find::Find;
use nvcc::Nvcc;
use std::env;
use std::path::PathBuf;
use std::vec::Vec;

const NVCC_SEARCH_LINUX: &[&str] = &["/usr/local/cuda/bin", "/usr/local/cuda*/bin"];

pub fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=runtime.h");
    println!("cargo:rerun-if-changed=driver.h");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Try to use NVCC to get the cuda include and library paths
    let nvcc = match Find::new("nvcc")
        .search_env("NVCC_PATH")
        .search_globs(NVCC_SEARCH_LINUX)
        .execute()
    {
        Ok(path) => Nvcc::new(path).unwrap(),
        Err(message) => panic!(message),
    };

    let include_flags = nvcc.include_flags();
    eprintln!("nvcc include flags: {:?}", include_flags);

    // Build runtime bindings
    bindgen::Builder::default()
    // .trust_clang_mangling(false)
    .header("runtime.h")
    .clang_args(include_flags)
    .opaque_type("max_align_t")
    .generate()
    .expect("Unable to generate bindings")
    .write_to_file(out_path.join("runtime.rs"))
    .expect("coudn't write to file");

    // Build driver bindings
    bindgen::Builder::default()
    // .trust_clang_mangling(false)
    .header("driver.h")
    .clang_args(include_flags)
    .opaque_type("max_align_t")
    .generate()
    .expect("Unable to generate bindings")
    .write_to_file(out_path.join("driver.rs"))
    .expect("coudn't write to file");

    // Try to find the CUDA libraries
    let cudart_path = match Find::new("libcudart.so.*")
        .search_env("LIBCUDART_PATH")
        .search_globs(
            &nvcc.libraries_paths()
                .iter()
                .map(|p| p.to_str().unwrap())
                .collect::<Vec<_>>(),
        )
        .execute()
    {
        Ok(path) => path,
        Err(message) => panic!(message),
    };

    eprintln!("Found cudart: {:?}", cudart_path);

    // Discover the version of the CUDA libraries
    // Set the corresponding feature flags
    if let Some(version) = find::parse_version(&cudart_path) {
        if version.len() > 0 {
            match version[0] {
                9 => {
                    println!("cargo:rustc-cfg=gte_cuda_9");
                }
                _ => (),
            }
        }
    }

    // Emit the link commands
    for path in nvcc.libraries_paths() {
        println!(
            "cargo:rustc-link-search=native={}",
            path.to_str().unwrap()
        );
    }
    println!("cargo:rustc-link-lib=cudart");
}
