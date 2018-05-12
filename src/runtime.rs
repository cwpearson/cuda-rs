#[allow(non_camel_case_types)]

use libc::{c_int, size_t};
use std::error;
use std::fmt;
use std::result;
use std::mem;

extern "C" {
    fn cudaGetDeviceCount(count: *mut c_int) -> cudaError;
    fn cudaMemGetInfo(free: *mut size_t, total: *mut size_t) -> cudaError;
    fn cudaDeviceReset() -> cudaError;
    fn cudaSetDevice(dev: c_int) -> cudaError;
}

#[repr(u32)]
#[derive(PartialEq, Debug, Clone)]
pub enum cudaError {
    cudaSuccess = 0,
    cudaErrorInitializationError = 3,
    cudaErrorLaunchFailure = 4,
    cudaErrorInvalidValue = 11,
}

impl fmt::Display for cudaError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let printable = match self {
            _ => stringify!(self),
        };
        write!(f, "{}", printable)
    }
}

#[derive(Debug, Clone)]
pub struct Error {
    cuda_error: cudaError,
}

impl Error {
    pub fn new(cuda_error: cudaError) -> Error {
        Error {
            cuda_error: cuda_error,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "cuda_error: {}", self.cuda_error)
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        "cuda_error_description"
    }
}

pub type Result<T> = result::Result<T, Error>;

pub fn device_count() -> Result<i32> {
    let mut count = 0 as c_int;
    let err = unsafe { cudaGetDeviceCount(&mut count as *mut c_int) };
    if err == cudaError::cudaSuccess {
        return Ok(count);
    } else {
        return Err(Error::new(err));
    }
}