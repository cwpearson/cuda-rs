extern crate find;
extern crate glob;

use std::path::PathBuf;
use std::path::Path;
use std::vec::Vec;
use glob::MatchOptions;
use std::env;
use find::Find;

const SEARCH_LINUX: &[&str] = &["/usr/local/cuda/lib*", "/usr/local/cuda*/lib*"];

pub fn main() {
    // Try to use NVCC to get the cuda library paths

    // Try to find the CUDA libraries

    let cudart_path = match Find::new("libcudart.so.*")
        .search_env("LIBCUDA_PATH")
        .search_globs(SEARCH_LINUX)
        .execute()
    {
        Ok(path) => path,
        Err(message) => panic!(message),
    };

    let lib_path = cudart_path.parent().unwrap();
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
    println!(
        "cargo:rustc-link-search=native={}",
        lib_path.to_str().unwrap()
    );
    println!("cargo:rustc-link-lib=cudart");
}
