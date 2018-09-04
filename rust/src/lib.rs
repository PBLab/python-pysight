extern crate bitreader;
extern crate libc;
#[macro_use] extern crate failure;
#[macro_use] extern crate failure_derive;
extern crate filebuffer;
extern crate data_encoding;

pub mod reading;
use reading::{analyze_lst, DataLine};

use libc::{uint64_t, c_char, c_void, size_t};
use std::ffi::CStr;
use std::{mem, slice};

/// Used to hold pointers to Vectors contains the analyzed data across FFI boundaries,
#[repr(C)]
pub struct DataPerChannel {
    ch_1: VecSlice,
    ch_2: VecSlice,
    ch_3: VecSlice,
    ch_4: VecSlice,
    ch_5: VecSlice,
    ch_6: VecSlice,
}

impl DataPerChannel {
    fn new(ch_1: VecSlice, ch_2: VecSlice, ch_3: VecSlice, ch_4: VecSlice, ch_5: VecSlice, ch_6: VecSlice)
        -> DataPerChannel {
        DataPerChannel { ch_1, ch_2, ch_3, ch_4, ch_5, ch_6 }
    }
}

/// A pair of pointer and data length to act as a vector in an FFI boundary.
#[repr(C)]
pub struct VecSlice {
    ptr: *mut u64,
    len: u64,
}

impl VecSlice {
    fn new(ptr: *mut u64, len: u64) -> VecSlice {
        VecSlice { ptr, len }
    }
}

/// Convert the input from Python into Rust data structures and call the main function
/// that reads and analyzes the `.lst` file
#[no_mangle]
pub extern "C" fn read_lst(file_path_py: *const c_char, start_of_data_pos: uint64_t,
                           range: uint64_t, timepatch_py: *const c_char) {
    
    let file_path_unsafe = unsafe {
        assert!(!file_path_py.is_null());
        CStr::from_ptr(file_path_py)
    };
    let file_path = file_path_unsafe.to_str().unwrap();

    let timepatch_unsafe = unsafe {
        assert!(!timepatch_py.is_null());
        CStr::from_ptr(timepatch_py)
    };
    let timepatch = timepatch_unsafe.to_str().unwrap();

    println!("{}, {}, {}", file_path, timepatch, range);


    // let s1 = VecSlice::new(p1, len1 };
    // ///
    // DataPerChannel::new(s1, s2, s3, s4, s5, s6)
}
