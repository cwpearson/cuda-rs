extern crate libc;

pub mod driver;
pub mod runtime;

use std::mem;
use std::result;

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
    Unknown,
}

impl ErrorKind {
    fn from_runtime(err: runtime::cudaError) -> ErrorKind {
        use runtime::*;
        match err {
            cudaError_cudaSuccess => ErrorKind::Success,
            cudaError_cudaErrorUnknown => ErrorKind::Unknown,
            _ => panic!("Unhandled runtime error {}", err),
        }
    }

    fn from_driver(err: runtime::cudaError_t) -> ErrorKind {
        match err {
            _ => panic!("Unhandled driver error {}", err),
        }
    }
}

impl From<runtime::cudaError> for Error {
    fn from(err: runtime::cudaError) -> Self {
        Error {
            kind: ErrorKind::from_runtime(err),
        }
    }
}

pub type Result<T> = result::Result<T, Error>;

pub fn device_count() -> Result<i64> {
    let mut count: libc::c_int = unsafe { mem::uninitialized() };
    let err = unsafe { runtime::cudaGetDeviceCount(&mut count) };
    match err {
        runtime::cudaError_cudaSuccess => Ok(count as i64),
        _ => Err(Error::from(err)),
    }
}
