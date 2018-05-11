extern crate glob;

use std::path::PathBuf;
use std::path::Path;
use std::vec::Vec;
use glob::MatchOptions;
use std::env;

const SEARCH_LINUX: &[&str] = &[
    "/usr/local/cuda/lib",
    "/usr/local/cuda*/lib"
];

fn find_version(file: &str) -> Option<&str> {
    if file.starts_with("libcudart.so.") {
        Some(&file[13..])
    } else {
        None
    }
}

/// Returns the components of the version appended to the supplied file.
fn parse_version(file: &Path) -> Vec<u32> {
    let file = file.file_name().and_then(|f| f.to_str()).unwrap_or("");
    let version = find_version(file).unwrap_or("");
    version
        .split('.')
        .map(|s| s.parse::<u32>().unwrap_or(0))
        .collect()
}

/// Returns a path to one of the supplied files if such a file can be found in the supplied directory.
fn contains(directory: &Path, files: &[String]) -> Option<PathBuf> {
    // Join the directory to the files to obtain our glob patterns.
    let patterns = files
        .iter()
        .filter_map(|f| directory.join(f).to_str().map(ToOwned::to_owned));

    // Prevent wildcards from matching path separators.
    let mut options = MatchOptions::new();
    options.require_literal_separator = true;

    // Collect any files that match the glob patterns.
    let mut matches = patterns
        .flat_map(|p| {
            if let Ok(paths) = glob::glob_with(&p, &options) {
                paths.filter_map(Result::ok).collect()
            } else {
                vec![]
            }
        })
        .collect::<Vec<_>>();

    // Sort the matches by their version, preferring shorter and higher versions.
    matches.sort_by_key(|m| parse_version(m));
    matches.pop()
}

fn find(files: &[String], env: &str) -> Result<PathBuf, String> {
    /// Searches the supplied directory and, on Windows, any relevant sibling directories.
    macro_rules! search_directory {
        ($directory: ident) => {
            if let Some(file) = contains(&$directory, files) {
                return Ok(file);
            }

            // On Windows, `libclang.dll` is usually found in the LLVM `bin` directory while
            // `libclang.lib` is usually found in the LLVM `lib` directory. To keep things
            // consistent with other platforms, only LLVM `lib` directories are included in the
            // backup search directory globs so we need to search the LLVM `bin` directory here.
            if cfg!(target_os = "windows") && $directory.ends_with("lib") {
                let sibling = $directory.parent().unwrap().join("bin");
                if let Some(file) = contains(&sibling, files) {
                    return Ok(file);
                }
            }
        };
    }

    // Search the directory provided by the relevant environment variable if it is set.
    if let Ok(directory) = env::var(env).map(|d| Path::new(&d).to_path_buf()) {
        search_directory!(directory);
    }

    // Search the `LD_LIBRARY_PATH` directories.
    if let Ok(path) = env::var("LD_LIBRARY_PATH") {
        for directory in path.split(":").map(Path::new) {
            search_directory!(directory);
        }
    }

    // Search the backup directories.
    let search = if cfg!(any(target_os = "freebsd", target_os = "linux")) {
        SEARCH_LINUX
    } else {
        &[]
    };
    for pattern in search {
        eprintln!("Searching for {}", pattern);
        let mut options = MatchOptions::new();
        options.case_sensitive = false;
        options.require_literal_separator = true;
        if let Ok(paths) = glob::glob_with(pattern, &options) {
            for path in paths.filter_map(Result::ok).filter(|p| p.is_dir()) {
                eprintln!("Looking in {:?}", path);
                search_directory!(path);
            }
        }
    }

    let message =
        format!(
        "couldn't find any of [{}], set the {} environment variable to a path where one of these \
         files can be found",
        files.iter().map(|f| format!("'{}'", f)).collect::<Vec<_>>().join(", "),
        env,
    );
    Err(message)
}

pub fn main() {

// Try to use NVCC to get the cuda library paths

// Try to find the CUDA libraries
let cudart_path = match find(&["libcudart.so.*".to_string()], "LIBCUDA_PATH") {
    Ok(path) => path,
    Err(message) => panic!(message),
};
let lib_path = cudart_path.parent().unwrap();
eprintln!("Found cudart: {:?}", cudart_path);

// Discover the version of the CUDA libraries
let version = parse_version(&cudart_path);

// Set the corresponding feature flags
if version.len() > 0 {
    match version[0] {
        9 => {
            println!("cargo:rustc-cfg=gte_cuda_9");
        }
        _ => (),
    }
}

// Emit the link commands
println!("cargo:rustc-link-search=native={}",
        lib_path.to_str().unwrap());
println!("cargo:rustc-link-lib=cudart");

}