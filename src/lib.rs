extern crate libc;
#[macro_use]
extern crate lazy_static;


mod driver;
mod runtime;

use std::result;
use std::mem;

lazy_static! {
    static ref SYSTEM: System = System::new();
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

struct Error {}

type Result<T> = result::Result<T, Error>;

pub fn device(id: i32) -> Option<Device> {
    if let Ok(count) = runtime::device_count() {
        if id < count {
            Some(Device::new(id))
        } else {
            None
        }
    } else {
        None
    }
}

pub struct System {
    devices: Vec<Device>,
}

impl System {
    fn new() -> System {
        let num_devs = runtime::device_count().unwrap();
        let mut devs = vec![];
        for i in 0..num_devs {
            devs.push(Device::new(i));
        }

        System {
            devices: devs,
        }
    }
}

pub struct Device {
    id: i32,
}

impl Device {
    fn new(id: i32) -> Device {
        Device {
            id: id
        }
    }
}
