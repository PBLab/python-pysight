use rayon::prelude::*;
use std::str;
use std::sync::Mutex;
use rayon::prelude::*;
use byteorder::{ReadBytesExt, LE};
use pyo3::prelude::*;
use pyo3::{ToPyObject, IntoPyObject, PyObject, PyDict};


use crate::helper_funcs::*;

pub const NUM_OF_INPUT_CHANNELS: usize = 6;

/// An array with four entries:
/// 0 - data lost
/// 1 - number of tag bits
/// 2 - number of sweep bits
/// 3 - number of time bits
pub struct TimepatchBits;

impl TimepatchBits {
    pub fn new(tp: &str) -> [u8; 4] {
        match tp {
            "0" => [0, 0, 0, 12],
            "5" => [0, 0, 8, 20],
            "1" => [0, 0, 0, 28],
            "1a" => [0, 0, 16, 28],
            "2a" => [0, 8, 8, 28],
            "22" => [0, 8, 0, 36],
            "32" => [1, 0, 7, 36],
            "2" => [0, 0, 0, 44],
            "5b" => [1, 15, 16, 28],
            "Db" => [0, 16, 16, 28],
            "f3" => [1, 16, 7, 36],
            "43" => [1, 15, 0, 44],
            "c3" => [0, 16, 0, 44],
            "3" => [1, 5, 0, 54],
            _ => panic!("Invalid timepatch value: {}.", tp),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DataLineU8 {
    lost: Vec<bool>,
    tag: Vec<u8>,
    edge: Vec<bool>,
    time: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct DataLineU16 {
    lost: Vec<bool>,
    tag: Vec<u16>,
    edge: Vec<bool>,
    time: Vec<u64>,
}

impl DataLineU8 {
    pub fn new(lost: Vec<bool>, tag: Vec<u8>, edge: Vec<bool>, time: Vec<u64>) -> DataLineU8 {
        DataLineU8 {lost, tag, edge, time}
    }

    pub fn push_lost(&mut self, val: bool) {
        &mut self.lost.push(val);
    }

    pub fn push_edge(&mut self, val: bool) {
        &mut self.edge.push(val);
    }

    pub fn push_tag(&mut self, val: u8) {
        &mut self.tag.push(val);
    }

    pub fn push_time(&mut self, val: u64) {
        &mut self.time.push(val);
    }
}

impl IntoPyObject for DataLineU8 {
    fn into_object(self, py: Python) -> PyObject {
        let dict = PyDict::new(py);
        dict.set_item("lost", self.lost).expect("Lost insertion error.");
        dict.set_item("tag", self.tag).expect("TAG insertion error.");
        dict.set_item("edge", self.edge).expect("Edge insertion error.");
        dict.set_item("time", self.time).expect("Time insertion error.");

        dict.into()
    }
}

impl ToPyObject for DataLineU8 {
    fn to_object(&self, py: Python) -> PyObject {
        let dict = PyDict::new(py);
        dict.set_item("lost", &self.lost).expect("Lost insertion error.");
        dict.set_item("tag", &self.tag).expect("TAG insertion error.");
        dict.set_item("edge", &self.edge).expect("Edge insertion error.");
        dict.set_item("time", &self.time).expect("Time insertion error.");

        dict.into()
    }
}

impl DataLineU16 {
    pub fn new(lost: Vec<bool>, tag: Vec<u16>, edge: Vec<bool>, time: Vec<u64>) -> DataLineU16 {
        DataLineU16 {lost, tag, edge, time}
    }

    pub fn push_lost(&mut self, val: bool) {
        &mut self.lost.push(val);
    }

    pub fn push_edge(&mut self, val: bool) {
        &mut self.edge.push(val);
    }

    pub fn push_tag(&mut self, val: u16) {
        &mut self.tag.push(val);
    }

    pub fn push_time(&mut self, val: u64) {
        &mut self.time.push(val);
    }
}

impl IntoPyObject for DataLineU16 {
    fn into_object(self, py: Python) -> PyObject {
        let dict = PyDict::new(py);
        dict.set_item("lost", self.lost).expect("Lost insertion error.");
        dict.set_item("tag", self.tag).expect("TAG insertion error.");
        dict.set_item("edge", self.edge).expect("Edge insertion error.");
        dict.set_item("time", self.time).expect("Time insertion error.");

        dict.into()
    }
}

impl ToPyObject for DataLineU16 {
    fn to_object(&self, py: Python) -> PyObject {
        let dict = PyDict::new(py);
        dict.set_item("lost", &self.lost).expect("Lost insertion error.");
        dict.set_item("tag", &self.tag).expect("TAG insertion error.");
        dict.set_item("edge", &self.edge).expect("Edge insertion error.");
        dict.set_item("time", &self.time).expect("Time insertion error.");

        dict.into()
    }
}

#[derive(Debug)]
pub struct LstReturnU16 {
    stop1: DataLineU16,
    stop2: DataLineU16,
    stop3: DataLineU16,
    stop4: DataLineU16,
    stop5: DataLineU16,
    start: DataLineU16,
}

impl LstReturnU16 {
    pub fn new(stop1: DataLineU16, stop2: DataLineU16, stop3: DataLineU16,
               stop4: DataLineU16, stop5: DataLineU16, start: DataLineU16) -> Self {
                   LstReturnU16 { stop1, stop2, stop3, stop4, stop5, start }
               }
}

impl IntoPyObject for LstReturnU16 {
    fn into_object(self, py: Python) -> PyObject {
        let dict = PyDict::new(py);
        dict.set_item("stop1", self.stop1).expect("STOP1 error");
        dict.set_item("stop2", self.stop2).expect("STOP2 error");
        dict.set_item("stop3", self.stop3).expect("STOP3 error");
        dict.set_item("stop4", self.stop4).expect("STOP4 error");
        dict.set_item("stop5", self.stop5).expect("STOP5 error");
        dict.set_item("start", self.start).expect("START error");
        dict.into()
    }
}

#[derive(Debug)]
pub struct LstReturnU8 {
    stop1: DataLineU8,
    stop2: DataLineU8,
    stop3: DataLineU8,
    stop4: DataLineU8,
    stop5: DataLineU8,
    start: DataLineU8,
}

impl LstReturnU8 {
    pub fn new(stop1: DataLineU8, stop2: DataLineU8, stop3: DataLineU8,
               stop4: DataLineU8, stop5: DataLineU8, start: DataLineU8) -> Self {
                   LstReturnU8 { stop1, stop2, stop3, stop4, stop5, start }
               }
}

impl IntoPyObject for LstReturnU8 {
    fn into_object(self, py: Python) -> PyObject {
        let dict = PyDict::new(py);
        dict.set_item("stop1", self.stop1).expect("STOP1 error");
        dict.set_item("stop2", self.stop2).expect("STOP2 error");
        dict.set_item("stop3", self.stop3).expect("STOP3 error");
        dict.set_item("stop4", self.stop4).expect("STOP4 error");
        dict.set_item("stop5", self.stop5).expect("STOP5 error");
        dict.set_item("start", self.start).expect("START error");
        dict.into()
    }
}


/// Parse a list file for time patch "1"
pub fn parse_1(data: &[u8], _range: u64, bit_order: &[u8; 4],
               parsed_data: Vec<Mutex<DataLineU16>>) 
    -> LstReturnU16 {
    let bitmap = to_bits_u32(bit_order);
    data
        .par_chunks(4)
        .filter_map(|mut line| if line != [0u8; 4] { 
            line.read_u32::<LE>().ok()
            } else { None })
        .map(|mut line| {
            println!("{:?}", line);
            let ch = ((line & 0b111) - 1) as usize;
            line = line >> 3;  // throw away "channel" bits
            let edge = (line & 0b1) == 1;
            line = line >> 1;  // throw away "edge" bit
            let time: u64 = (line & bitmap[3]).into();
            let mut dataline = parsed_data[ch].lock().expect("Mutex lock error");
            dataline.push_edge(edge);
            dataline.push_time(time);
        }).collect::<Vec<_>>();
    let mut parsed_data_no_mutex: Vec<DataLineU16> = parsed_data.into_iter().map(|x| x.into_inner().unwrap()).collect();
    LstReturnU16::new(parsed_data_no_mutex.remove(0),
                     parsed_data_no_mutex.remove(0),
                     parsed_data_no_mutex.remove(0),
                     parsed_data_no_mutex.remove(0),
                     parsed_data_no_mutex.remove(0),
                     parsed_data_no_mutex.remove(0))
}

/// Parse a list file for time patch "0"
pub fn parse_0(data: &[u8], _range: u64, bit_order: &[u8; 4],
               parsed_data: Vec<Mutex<DataLineU16>>) -> LstReturnU16 {
    let bitmap = to_bits_u16(bit_order);
    data
        .par_chunks(2)
        .filter_map(|mut line| if line != [0u8; 2] { 
            line.read_u16::<LE>().ok()
            } else { None })
        .map(|mut line| {
            println!("{:?}", line);
            let ch = ((line & 0b111) - 1) as usize;
            line = line >> 3;  // throw away "channel" bits
            let edge = (line & 0b1) == 1;
            line = line >> 1;  // throw away "edge" bit
            let time: u64 = (line & bitmap[3]).into();
            let mut dataline = parsed_data[ch].lock().expect("Mutex lock error");
            dataline.push_edge(edge);
            dataline.push_time(time);
        }).collect::<Vec<_>>();
    let mut parsed_data_no_mutex: Vec<DataLineU16> = parsed_data.into_iter().map(|x| x.into_inner().unwrap()).collect();
    LstReturnU16::new(parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0))
}

/// Parse a list file for time patch "5"
pub fn parse_5(data: &[u8], range: u64, bit_order: &[u8; 4],
               parsed_data: Vec<Mutex<DataLineU16>>) 
    -> LstReturnU16 {
    let bitmap = to_bits_u32(bit_order);
    data
        .par_chunks(4)
        .filter_map(|mut line| if line != [0u8; 4] { 
            line.read_u32::<LE>().ok()
            } else { None })
        .map(|mut line| {
            println!("line: {:b}", line);
            let ch = ((line & 0b111) - 1) as usize;
            line = line >> 3;  // throw away "channel" bits
            let edge = (line & 0b1) == 1;
            line = line >> 1;  // throw away "edge" bit
            let mut time: u64 = (line & bitmap[3]) as u64;
            line = line >> bit_order[3]; // throw away "time" bits
            let sweep: u8 = (line & bitmap[2]) as u8;
            time += range * (u64::from(sweep - 1));
            let mut dataline = parsed_data[ch].lock().expect("Mutex lock error");
            dataline.push_edge(edge);
            dataline.push_time(time);
        }).collect::<Vec<_>>();
    let mut parsed_data_no_mutex: Vec<DataLineU16> = parsed_data.into_iter().map(|x| x.into_inner().unwrap()).collect();
    LstReturnU16::new(parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0))
}


/// Parse a list file for time patch "1a"
pub fn parse_1a(data: &[u8], range: u64, bit_order: &[u8; 4],
                parsed_data: Vec<Mutex<DataLineU16>>) 
    -> LstReturnU16 {
    let bitmap = to_bits_u64(bit_order);
    data
        .par_chunks(6)
        .filter_map(|mut line| if line != [0u8; 6] { 
            line.read_u48::<LE>().ok()
            } else { None })
        .map(|mut line| {
            let ch = ((line & 0b111) - 1) as usize;
            line = line >> 3;  // throw away "channel" bits
            let edge = (line & 0b1) == 1;
            line = line >> 1;  // throw away "edge" bit
            let mut time: u64 = (line & bitmap[3]) as u64;
            line = line >> bit_order[3]; // throw away "time" bits
            let sweep: u16 = (line & bitmap[2]) as u16;
            time += range * (u64::from(sweep - 1));
            let mut dataline = parsed_data[ch].lock().expect("Mutex lock error");
            dataline.push_edge(edge);
            dataline.push_time(time);
        }).collect::<Vec<_>>();
    let mut parsed_data_no_mutex: Vec<DataLineU16> = parsed_data.into_iter().map(|x| x.into_inner().unwrap()).collect();
    LstReturnU16::new(parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0))}


/// Parse a list file for time patch "2a"
pub fn parse_2a(data: &[u8], range: u64, bit_order: &[u8; 4],
                parsed_data: Vec<Mutex<DataLineU8>>) -> LstReturnU8 {
    let bitmap = to_bits_u64(bit_order);
    data
        .par_chunks(6)
        .filter_map(|mut line| if line != [0u8; 6] { 
            line.read_u48::<LE>().ok()
            } else { None })
        .map(|mut line| {
            let ch = ((line & 0b111) - 1) as usize;
            line = line >> 3;  // throw away "channel" bits
            let edge = (line & 0b1) == 1;
            line = line >> 1;  // throw away "edge" bit
            let mut time: u64 = (line & bitmap[3]) as u64;
            line = line >> bit_order[3]; // throw away "time" bits
            let sweep: u16 = (line & bitmap[2]) as u16;
            time += range * (u64::from(sweep - 1));
            line = line >> bit_order[2]; // throw away "sweep" bits
            let tag: u8 = (line & bitmap[1]) as u8;
            let mut dataline = parsed_data[ch].lock().expect("Mutex lock error");
            dataline.push_edge(edge);
            dataline.push_time(time);
            dataline.push_tag(tag);
        }).collect::<Vec<_>>();
    let mut parsed_data_no_mutex: Vec<DataLineU8> = parsed_data.into_iter().map(|x| x.into_inner().unwrap()).collect();
    LstReturnU8::new(parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0))}


/// Parse a list file for time patch "22"
pub fn parse_22(data: &[u8], _range: u64, bit_order: &[u8; 4],
                parsed_data: Vec<Mutex<DataLineU8>>) -> LstReturnU8 {
    let bitmap = to_bits_u64(bit_order);
    data
        .par_chunks(6)
        .filter_map(|mut line| if line != [0u8; 6] { 
            line.read_u48::<LE>().ok()
            } else { None })
        .map(|mut line| {
            let ch = ((line & 0b111) - 1) as usize;
            line = line >> 3;  // throw away "channel" bits
            let edge = (line & 0b1) == 1;
            line = line >> 1;  // throw away "edge" bit
            let time: u64 = (line & bitmap[3]) as u64;
            line = line >> bit_order[3]; // throw away "time" bits
            let tag: u8 = (line & bitmap[1]) as u8;
            let mut dataline = parsed_data[ch].lock().expect("Mutex lock error");
            dataline.push_edge(edge);
            dataline.push_time(time);
            dataline.push_tag(tag);
        }).collect::<Vec<_>>();
    let mut parsed_data_no_mutex: Vec<DataLineU8> = parsed_data.into_iter().map(|x| x.into_inner().unwrap()).collect();
    LstReturnU8::new(parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0))
}


/// Parse a list file for time patch "32"
pub fn parse_32(data: &[u8], range: u64, bit_order: &[u8; 4],
                parsed_data: Vec<Mutex<DataLineU16>>) -> LstReturnU16 {
    let bitmap = to_bits_u64(bit_order);
    data
        .par_chunks(6)
        .filter_map(|mut line| if line != [0u8; 6] { 
            line.read_u48::<LE>().ok()
            } else { None })
        .map(|mut line| {
            let ch = ((line & 0b111) - 1) as usize;
            line = line >> 3;  // throw away "channel" bits
            let edge = (line & 0b1) == 1;
            line = line >> 1;  // throw away "edge" bit
            let mut time: u64 = (line & bitmap[3]) as u64;
            line = line >> bit_order[3]; // throw away "time" bits
            let sweep: u16 = (line & bitmap[2]) as u16;
            time += range * (u64::from(sweep - 1));
            line = line >> bit_order[2]; // throw away "sweep" bits
            let lost: bool = (line & bitmap[0]) == 1;
            let mut dataline = parsed_data[ch].lock().expect("Mutex lock error");
            dataline.push_edge(edge);
            dataline.push_time(time);
            dataline.push_lost(lost);
        }).collect::<Vec<_>>();
    let mut parsed_data_no_mutex: Vec<DataLineU16> = parsed_data.into_iter().map(|x| x.into_inner().unwrap()).collect();
    LstReturnU16::new(parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0))
}


/// Parse a list file for time patch "2"
pub fn parse_2(data: &[u8], _range: u64, bit_order: &[u8; 4],
               parsed_data: Vec<Mutex<DataLineU16>>) -> LstReturnU16 {
    let bitmap = to_bits_u64(bit_order);
    data
        .par_chunks(6)
        .filter_map(|mut line| if line != [0u8; 6] { 
            line.read_u48::<LE>().ok()
            } else { None })
        .map(|mut line| {
            let ch = ((line & 0b111) - 1) as usize;
            line = line >> 3;  // throw away "channel" bits
            let edge = (line & 0b1) == 1;
            line = line >> 1;  // throw away "edge" bit
            let time: u64 = (line & bitmap[3]) as u64;
            let mut dataline = parsed_data[ch].lock().expect("Mutex lock error");
            dataline.push_edge(edge);
            dataline.push_time(time);
        }).collect::<Vec<_>>();
    let mut parsed_data_no_mutex: Vec<DataLineU16> = parsed_data.into_iter().map(|x| x.into_inner().unwrap()).collect();
    LstReturnU16::new(parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0))
}


/// Parse a list file for time patch "5b"
pub fn parse_5b(data: &[u8], range: u64, bit_order: &[u8; 4],
                parsed_data: Vec<Mutex<DataLineU16>>) 
    -> LstReturnU16 {
    let bitmap = to_bits_u64(bit_order);
    data
        .par_chunks(8)
        .filter_map(|mut line| if line != [0u8; 8] { 
            line.read_u64::<LE>().ok()
            } else { None })
        .map(|mut line| {
            let ch = ((line & 0b111) - 1) as usize;
            line = line >> 3;  // throw away "channel" bits
            let edge = (line & 0b1) == 1;
            line = line >> 1;  // throw away "edge" bit
            let mut time: u64 = (line & bitmap[3]) as u64;
            line = line >> bit_order[3]; // throw away "time" bits
            let sweep: u16 = (line & bitmap[2]) as u16;
            time += range * (u64::from(sweep - 1));
            line = line >> bit_order[2]; // throw away "sweep" bits
            let tag: u16 = (line & bitmap[1]) as u16;
            line = line >> bit_order[1]; // throw away "tag" bits
            let lost: bool = (line & bitmap[0]) == 1;
            let mut dataline = parsed_data[ch].lock().expect("Mutex lock error");
            dataline.push_edge(edge);
            dataline.push_time(time);
            dataline.push_tag(tag);
            dataline.push_lost(lost);
        }).collect::<Vec<_>>();
    let mut parsed_data_no_mutex: Vec<DataLineU16> = parsed_data.into_iter().map(|x| x.into_inner().unwrap()).collect();
    LstReturnU16::new(parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0))
}

/// Parse a list file for time patch "Db"
pub fn parse_Db(data: &[u8], range: u64, bit_order: &[u8; 4],
                parsed_data: Vec<Mutex<DataLineU16>>) -> LstReturnU16 {
    let bitmap = to_bits_u64(bit_order);
    data
        .par_chunks(8)
        .filter_map(|mut line| if line != [0u8; 8] { 
            line.read_u64::<LE>().ok()
            } else { None })
        .map(|mut line| {
            let ch = ((line & 0b111) - 1) as usize;
            line = line >> 3;  // throw away "channel" bits
            let edge = (line & 0b1) == 1;
            line = line >> 1;  // throw away "edge" bit
            let mut time: u64 = (line & bitmap[3]) as u64;
            line = line >> bit_order[3]; // throw away "time" bits
            let sweep: u16 = (line & bitmap[2]) as u16;
            time += range * (u64::from(sweep - 1));
            line = line >> bit_order[2];  // throw away "sweep bits"
            let tag: u16 = (line & bitmap[1]) as u16;
            let mut dataline = parsed_data[ch].lock().expect("Mutex lock error");
            dataline.push_edge(edge);
            dataline.push_time(time);
            dataline.push_tag(tag);
        }).collect::<Vec<_>>();
    let mut parsed_data_no_mutex: Vec<DataLineU16> = parsed_data.into_iter().map(|x| x.into_inner().unwrap()).collect();
    LstReturnU16::new(parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0))
}

/// Parse a list file for time patch "f3"
pub fn parse_f3(data: &[u8], range: u64, bit_order: &[u8; 4],
                parsed_data: Vec<Mutex<DataLineU16>>) -> LstReturnU16 { 
    let bitmap = to_bits_u64(bit_order);
    data
        .par_chunks(8)
        .filter_map(|mut line| if line != [0u8; 8] { 
            line.read_u64::<LE>().ok()
            } else { None })
        .map(|mut line| {
            let ch = ((line & 0b111) - 1) as usize;
            line = line >> 3;  // throw away "channel" bits
            let edge = (line & 0b1) == 1;
            line = line >> 1;  // throw away "edge" bit
            let mut time: u64 = (line & bitmap[3]) as u64;
            line = line >> bit_order[3]; // throw away "time" bits
            let sweep: u16 = (line & bitmap[2]) as u16;
            time += range * (u64::from(sweep - 1));
            line = line >> bit_order[2]; // throw away "sweep" bits
            let lost: bool = (line & bitmap[0]) == 1;
            line = line >> bit_order[0];  // throw away lost bit
            let tag: u16 = (line & bitmap[1]) as u16;
            let mut dataline = parsed_data[ch].lock().expect("Mutex lock error");
            dataline.push_edge(edge);
            dataline.push_time(time);
            dataline.push_tag(tag);
            dataline.push_lost(lost);
        }).collect::<Vec<_>>();
    let mut parsed_data_no_mutex: Vec<DataLineU16> = parsed_data.into_iter().map(|x| x.into_inner().unwrap()).collect();
    LstReturnU16::new(parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0))
}


/// Parse a list file for time patch "43"
pub fn parse_43(data: &[u8], _range: u64, bit_order: &[u8; 4],
                parsed_data: Vec<Mutex<DataLineU16>>) 
    -> LstReturnU16 {
    let bitmap = to_bits_u64(bit_order);
    data
        .par_chunks(8)
        .filter_map(|mut line| if line != [0u8; 8] { 
            line.read_u64::<LE>().ok()
            } else { None })
        .map(|mut line| {
            let ch = ((line & 0b111) - 1) as usize;
            line = line >> 3;  // throw away "channel" bits
            let edge = (line & 0b1) == 1;
            line = line >> 1;  // throw away "edge" bit
            let time: u64 = (line & bitmap[3]) as u64;
            line = line >> bit_order[3]; // throw away "time" bits
            let tag: u16 = (line & bitmap[1]) as u16;
            line = line >> bit_order[1]; // throw away "tag" bits
            let lost: bool = (line & bitmap[0]) == 1;
            let mut dataline = parsed_data[ch].lock().expect("Mutex lock error.");
            dataline.push_edge(edge);
            dataline.push_time(time);
            dataline.push_tag(tag);
            dataline.push_lost(lost);
        }).collect::<Vec<_>>();
    let mut parsed_data_no_mutex: Vec<DataLineU16> = parsed_data.into_iter().map(|x| x.into_inner().unwrap()).collect();
    LstReturnU16::new(parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0))
}


/// Parse a list file for time patch "c3"
pub fn parse_c3(data: &[u8], _range: u64, bit_order: &[u8; 4],
                parsed_data: Vec<Mutex<DataLineU16>>) -> LstReturnU16 { 
    let bitmap = to_bits_u64(bit_order);
    data
        .par_chunks(8)
        .filter_map(|mut line| if line != [0u8; 8] { 
            line.read_u64::<LE>().ok()
            } else { None })
        .map(|mut line| {
            let ch = ((line & 0b111) - 1) as usize;
            line = line >> 3;  // throw away "channel" bits
            let edge = (line & 0b1) == 1;
            line = line >> 1;  // throw away "edge" bit
            let time: u64 = (line & bitmap[3]) as u64;
            line = line >> bit_order[3]; // throw away "time" bits
            let tag: u16 = (line & bitmap[1]) as u16;
            let mut dataline = parsed_data[ch].lock().expect("Mutex lock error.");
            dataline.push_edge(edge);
            dataline.push_time(time);
            dataline.push_tag(tag);
        }).collect::<Vec<_>>();
    let mut parsed_data_no_mutex: Vec<DataLineU16> = parsed_data.into_iter().map(|x| x.into_inner().unwrap()).collect();
    LstReturnU16::new(parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0),
                      parsed_data_no_mutex.remove(0))
}

/// Parse a list file for time patch "3"
pub fn parse_3(data: &[u8], _range: u64, bit_order: &[u8; 4],
               parsed_data: Vec<Mutex<DataLineU8>>) 
    -> LstReturnU8 {
    let bitmap = to_bits_u64(bit_order);
    let _res: Vec<_> = data
        .par_chunks(8)
        .filter_map(|mut line| if line != [0u8; 8] { 
            line.read_u64::<LE>().ok()
            } else { None })
        .map(|mut line| {
            let ch = ((line & 0b111) - 1) as usize;
            line = line >> 3;  // throw away "channel" bits
            let edge = (line & 0b1) == 1;
            line = line >> 1;  // throw away "edge" bit
            let time: u64 = (line & bitmap[3]) as u64;
            line = line >> bit_order[3]; // throw away "time" bits
            let tag: u8 = (line & bitmap[1]) as u8;
            line = line >> bit_order[1]; // throw away "tag" bits
            let lost: bool = (line & bitmap[0]) == 1;
            let mut dataline = parsed_data[ch].lock().expect("Mutex lock error");
            dataline.push_edge(edge);
            dataline.push_time(time);
            dataline.push_tag(tag);
            dataline.push_lost(lost);
        }).collect();
    let mut parsed_data_no_mutex: Vec<DataLineU8> = parsed_data.into_iter().map(|x| x.into_inner().unwrap()).collect();
    LstReturnU8::new(parsed_data_no_mutex.remove(0),
                     parsed_data_no_mutex.remove(0),
                     parsed_data_no_mutex.remove(0),
                     parsed_data_no_mutex.remove(0),
                     parsed_data_no_mutex.remove(0),
                     parsed_data_no_mutex.remove(0))
}


// /// Parse a list file for time patch "5"
// fn seq_parse_5(data: &[u8], range: u64, bit_order: &[u8; 4],
//                            mut parsed_data: Vec<Vec<DataLine>>) 
//     -> Result<Vec<Vec<DataLine>>, Error> {
//     let num_of_bytes_per_line = ((bit_order.iter().sum::<u8>() + 4) / 8) as usize;
//     let bitmap = to_bits_u32(bit_order);
//     let res: Vec<_> = data
//         .chunks(4)
//         .filter_map(|mut line| if line != [0u8; 4] { 
//             line.read_u32::<LE>().ok()
//             } else { None })
//         .map(|mut line| {
//             let ch = ((line & 0b111) - 1) as usize;
//             line = line >> 3;  // throw away "channel" bits
//             let edge = (line & 0b1) == 1;
//             line = line >> 1;  // throw away "edge" bit
//             let mut time: u64 = (line & bitmap[3]) as u64;
//             line = line >> bit_order[3]; // throw away "time" bits
//             let sweep: u16 = (line & bitmap[2]) as u16;
//             time += range * (u64::from(sweep - 1));
//             line = line >> bit_order[2]; // throw away "sweep" bits
//             let tag: u16 = (line & bitmap[1]) as u16;
//             line = line >> bit_order[1]; // throw away "tag" bits
//             let lost: bool = (line & bitmap[0]) == 1;
//             let dl = DataLine::new(lost, tag, edge, time);
//             parsed_data[ch].push(dl);
//         }).collect();
//     Ok(parsed_data)
// }


/// Parse a list file for time patch "43"
// fn seq_parse_43(data: &[u8], range: u64, bit_order: &[u8; 4],
//                            mut parsed_data: Vec<Vec<DataLine>>) 
//     -> Result<Vec<Vec<DataLine>>, Error> {
//     let num_of_bytes_per_line = ((bit_order.iter().sum::<u8>() + 4) / 8) as usize;
//     let bitmap = to_bits_u64(bit_order);
//     let res: Vec<_> = data
//         .chunks(4)
//         .filter_map(|mut line| if line != [0u8; 4] { 
//             line.read_u64::<LE>().ok()
//             } else { None })
//         .map(|mut line| {
//             let ch = ((line & 0b111) - 1) as usize;
//             line = line >> 3;  // throw away "channel" bits
//             let edge = (line & 0b1) == 1;
//             line = line >> 1;  // throw away "edge" bit
//             let time: u64 = (line & bitmap[3]) as u64;
//             line = line >> bit_order[3]; // throw away "time" bits
//             line = line >> bit_order[2]; // throw away "sweep" bits
//             let tag: u16 = (line & bitmap[1]) as u16;
//             line = line >> bit_order[1]; // throw away "tag" bits
//             let lost: bool = (line & bitmap[0]) == 1;
//             let dl = DataLine::new(lost, tag, edge, time);
//             parsed_data[ch].push(dl);
//         }).collect();
//     Ok(parsed_data)
// }




/// Mock implementation of the parsing function that uses parallel
/// execution. Used for benchmarking.
#[cfg(test)]
pub fn analyze_lst_par(no: i32) -> Result<Vec<Mutex<Vec<DataLine>>>, Error> {
    let fname = "1000nm_Pulsatile_Modulation_-9000mV_to_9500mV_1_sweep_each_32s_long009.lst";  //002
    let start_of_data = 1568usize;  // 1565
    let range = 80000u64;
    let timepatch = "43";
    let channel_map = vec![1, 0, 0, 0, 0, 1];

    let data_with_headers = FileBuffer::open(fname).expect("bad file name");
    let data_size: usize = (fs::metadata(fname)?.len() - start_of_data as u64) as usize;
    let data = &data_with_headers[start_of_data..];
    // Open the file and convert it to a usable format

    let chan_map = create_channel_vec(data_size, channel_map);

    let tp_enum = Timepatch::new(timepatch);
    let processed_data = match tp_enum {
        Timepatch::Tp43(func) => func(data, range, &TimepatchBits::new(timepatch), chan_map),
        _ => panic!()
    };
    processed_data
}

/// Mock implementation of the parsing function that uses sequential,
/// instead of parallel, parsing. Used for benchmarking.
#[cfg(test)]
pub fn analyze_lst_seq(no: i32) -> Result<Vec<Vec<DataLine>>, Error> {
    
    let fname = "1000nm_Pulsatile_Modulation_-9000mV_to_9500mV_1_sweep_each_32s_long009.lst"; // 002
    let start_of_data = 1568usize;  // 1565
    let range = 80000u64;
    let timepatch = "43";
    let channel_map = vec![1, 0, 0, 0, 0, 1];

    let data_with_headers = FileBuffer::open(fname).expect("bad file name");
    let data_size: usize = (fs::metadata(fname)?.len() - start_of_data as u64) as usize;
    let data = &data_with_headers[start_of_data..];
    // Open the file and convert it to a usable format

    let chan_map = create_channel_vec_seq(data_size, channel_map.to_vec());

    let tp_enum = TimepatchBits::new(timepatch);
    let processed_data = parse_with_sweep_8bytes(data, range, &tp_enum, chan_map);
    processed_data

}
