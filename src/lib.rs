extern crate libc;

pub mod driver;
pub mod runtime;

use std::result;
use std::mem;


#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn cuda_test_dev0() {
        assert!(true);
    }
}

#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
}

#[derive(Debug)]
pub enum ErrorKind {
    Success,
}

impl ErrorKind {
    fn from_runtime(err: runtime::cudaError_t) -> ErrorKind {
        match err {
        runtime::cudaError_cudaSuccess => ErrorKind::Success,
        _ => panic!("Unhandled runtime error {}", err),
        }
    }

    fn from_driver(err: runtime::cudaError_t) -> ErrorKind {
        match err {
        _ => panic!("Unhandled driver error {}", err),
        }
    }
}

impl Error {
    fn from(err: runtime::cudaError) -> Error {
        Error {
            kind: ErrorKind::from_driver(err)
        }
    }
}

type Result<T> = result::Result<T, Error>;

pub fn device_count() -> Result<i64> {
    
    let mut count: libc::c_int = unsafe { mem::uninitialized() };
    let err = unsafe { runtime::cudaGetDeviceCount(&mut count) };
    match err {
        runtime::cudaError_cudaSuccess => Ok(count as i64),
        _ => Err(Error::from(err)),
    }
}
