use std::fs;
// use rayon::prelude::*;
use std::collections::HashMap;
use std::str;
use filebuffer::FileBuffer;
use bitreader::BitReader;
use failure::Error;
use data_encoding::HEXLOWER;

/// The basic struct with all of the information
/// contained in a single "line" of an .lst file
#[derive(Clone, Debug)]
#[repr(C)]
pub struct DataLine {
    lost: u8,
    tag: u16,
    edge: bool,
    sweep: u16,
    time: u64,
}

impl DataLine {
    fn new(lost: u8, tag: u16, edge: bool, sweep: u16, time: u64) -> DataLine {
        DataLine { lost, tag, edge, sweep, time }
    }
}

/// Generate a HashMap where the keys are the channel numbers, and the values
/// are vectors containing data
fn create_channel_map(data_size: usize, active_channels: Vec<u8>) -> HashMap<u8, Vec<DataLine>> {
    let mut channel_map = HashMap::new();

    let vec = Vec::with_capacity(data_size + 1);
//            vec.set_len(data_size + 1);
    for (idx, is_active) in active_channels.iter().enumerate() {
        if is_active == &1u8 {
            channel_map.insert((idx + 1) as u8, vec.clone());
        }
    }
    channel_map
}

/// Return type of our main function
type LstReturn = fn(&[u8], u64, &[u8; 4], HashMap<u8, Vec<DataLine>>)
        -> Result<HashMap<u8, Vec<DataLine>>, Error>;

/// Maps a timepatch value to the function that parses it.
enum Timepatch {
    Tp0(LstReturn),
    Tp5(LstReturn),
    Tp1(LstReturn),
    Tp1a(LstReturn),
    Tp2a(LstReturn),
    Tp22(LstReturn),
    Tp32(LstReturn),
    Tp2(LstReturn),
    Tp5b(LstReturn),
    TpDb(LstReturn),
    Tpf3(LstReturn),
    Tp43(LstReturn),
    Tpc3(LstReturn),
    Tp3(LstReturn),
}

impl Timepatch {
    fn new(tp: &str) -> Timepatch {
        match tp {
            "0" => Timepatch::Tp0(parse_no_sweep),
            "5" => Timepatch::Tp5(parse_with_sweep),
            "1" => Timepatch::Tp1(parse_no_sweep),
            "1a" => Timepatch::Tp1a(parse_with_sweep),
            "2a" => Timepatch::Tp2a(parse_with_sweep),
            "22" => Timepatch::Tp22(parse_no_sweep),
            "32" => Timepatch::Tp32(parse_with_sweep),
            "2" => Timepatch::Tp2(parse_no_sweep),
            "5b" => Timepatch::Tp5b(parse_with_sweep),
            "Db" => Timepatch::TpDb(parse_with_sweep),
            "f3" => Timepatch::Tpf3(parse_f3),
            "43" => Timepatch::Tp43(parse_no_sweep),
            "c3" => Timepatch::Tpc3(parse_no_sweep),
            "3" => Timepatch::Tp3(parse_no_sweep),
            _ => panic!("Invalid timepatch value: {}", tp),
        }
    }
}

struct TimepatchBits;

impl TimepatchBits {
    fn new(tp: &str) -> [u8; 4] {
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
            "Db" => [0, 0, 16, 28],
            "f3" => [1, 16, 7, 36],
            "43" => [1, 15, 0, 44],
            "c3" => [0, 16, 0, 44],
            "3" => [1, 5, 0, 54],
            _ => panic!("Invalid timepatch value: {}.", tp),
        }
    }
}

// pub fn old_main(file_path: String, start_of_data_pos: usize,
//                 range: u64, timepatch: String) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
//     let data_including_headers = FileBuffer::open(file_path.clone()).expect("File not found.");
//     println!("File opened successfully.");
//     let data_size: usize = (fs::metadata(file_path).expect("File not found.")
//                                                    .len() - start_of_data_pos as u64)
//                                                    as usize;

//     // Find the suitable bit order in the file
//     let timepatch_map = create_timepatch_map();
//     let bit_order: &[u8; 4] = timepatch_map.get(timepatch.as_str()).unwrap();
//     let mut chunk_size: u8 = bit_order.iter().sum();
//     chunk_size = (chunk_size + 4) / 8;

//     // Get data
//     let processed_data;
//     if String::from("f3") == timepatch {
//         processed_data = iterate_over_f3(&data_including_headers[start_of_data_pos..], range,
//                                          bit_order, chunk_size as usize, data_size);
//     } else {
//         processed_data = iterate_over_file(&data_including_headers[start_of_data_pos..], range,
//                                            bit_order, chunk_size as usize, data_size);
//     }

//     processed_data
// }

fn create_timepatch_map<'a>() -> HashMap<&'a str, [u8; 4]> {

    let mut timepatch_map = HashMap::new();

    // Array elements: Data lost, TAG bits, Sweep, Time (edge and channel are always 1 and 3)
    let array_for_32 = [1, 0, 7, 36];
    timepatch_map.insert("32", array_for_32);

    let array_for_f3 = [1, 16, 7, 36];
    timepatch_map.insert("f3", array_for_f3);

    let array_for_5b = [1, 15, 16, 28];
    timepatch_map.insert("5b", array_for_5b);

    timepatch_map
}

/// Parse data in file if timepatch == "f3"
fn parse_f3(data: &[u8], range: u64, bit_order: &[u8; 4],
            mut map_of_data: HashMap<u8, Vec<DataLine>>) -> Result<HashMap<u8, Vec<DataLine>>, Error> {
    let mut lost: u8;
    let mut tag: u16;
    let mut sweep: u16;
    let mut time: u64;
    let mut edge: bool;
    let mut chan: u8;

    let mut chunk_size: u8 = bit_order.iter().sum();
    chunk_size = (chunk_size  + 4u8) / 8u8;
    let mut reversed_vec = Vec::with_capacity(chunk_size as usize + 1);
    for cur_data in data.chunks(chunk_size as usize) {
        reversed_vec.truncate(0);
        reversed_vec.extend(cur_data.iter().rev());
        let mut reader = BitReader::new(&reversed_vec);
        tag = reader.read_u16(bit_order[1]).expect("(f3) tag read problem.");
        lost = reader.read_u8(bit_order[0]).expect("(f3) lost read problem.");
        sweep = reader.read_u16(bit_order[2]).expect("(f3) sweep read problem.");
        time = reader.read_u64(bit_order[3]).expect("(f3) time read problem");
        time += range * (u64::from(sweep - 1));
        edge = reader.read_bool().expect("(f3) edge read problem.");
        chan = reader.read_u8(3).expect("channel read problem.");


        // Populate a hashmap, each key being an input channel and the values are a vector
        // of DataLines
        map_of_data.get_mut(&chan).unwrap().push(DataLine::new(lost, tag, edge, sweep, time));
    }
    Ok(map_of_data)
}

/// Parse list files with a sweep counter
fn parse_with_sweep(data: &[u8], range: u64, bit_order: &[u8; 4],
                    mut map_of_data: HashMap<u8, Vec<DataLine>>) -> Result<HashMap<u8, Vec<DataLine>>, Error> {
    let mut lost: u8;
    let mut tag: u16;
    let mut sweep: u16;
    let mut time: u64;
    let mut edge: bool;
    let mut chan: u8;

    let mut chunk_size: u8 = bit_order.iter().sum();
    chunk_size = (chunk_size  + 4u8) / 8u8;
    let mut reversed_vec = Vec::with_capacity(chunk_size as usize + 1);
        for cur_data in data.chunks(chunk_size as usize) {
            reversed_vec.truncate(0);
            reversed_vec.extend(cur_data.iter().rev());
            let mut reader = BitReader::new(&reversed_vec);
            lost = reader.read_u8(bit_order[0]).expect("lost read problem.");
            tag = reader.read_u16(bit_order[1]).expect("tag read problem.");
            sweep = reader.read_u16(bit_order[2]).expect("sweep read problem.");
            time = reader.read_u64(bit_order[3]).expect("time read problem.");
            edge = reader.read_bool().expect("edge read problem.");
            chan = reader.read_u8(3).expect("channel read problem.");

            time += range * (u64::from(sweep - 1));
            // Populate a hashmap, each key being an input channel and the values are a vector
            // of DataLines
            map_of_data.get_mut(&chan).unwrap().push(DataLine::new(lost, tag, edge, sweep, time));
        }

    Ok(map_of_data)
}

/// Parse list files without a sweep counter
fn parse_no_sweep(data: &[u8], _range: u64, bit_order: &[u8; 4],
                  mut map_of_data: HashMap<u8, Vec<DataLine>>) -> Result<HashMap<u8, Vec<DataLine>>, Error> {
    let mut lost: u8;
    let mut tag: u16;
    let mut sweep: u16;
    let mut time: u64;
    let mut edge: bool;
    let mut chan: u8;


    let mut chunk_size: u8 = bit_order.iter().sum();
    chunk_size = (chunk_size  + 4u8) / 8u8;
    let mut reversed_vec = Vec::with_capacity(chunk_size as usize + 1);
        for cur_data in data.chunks(chunk_size as usize) {
            reversed_vec.truncate(0);
            reversed_vec.extend(cur_data.iter().rev());
            // reversed_vec.extend(cur_data.iter());
            let mut reader = BitReader::new(&reversed_vec);
            lost = reader.read_u8(bit_order[0]).expect("lost read problem.");
            tag = reader.read_u16(bit_order[1]).expect("tag read problem.");
            sweep = reader.read_u16(bit_order[2]).expect("sweep read problem.");
            time = reader.read_u64(bit_order[3]).expect("time read problem.");
            edge = reader.read_bool().expect("edge read problem.");
            chan = reader.read_u8(3).expect("channel read problem.");
            // Populate a hashmap, each key being an input channel and the values are a vector
            // of DataLines
            println!("{:?}", (lost, tag, time, edge, chan));
            map_of_data.get_mut(&chan)
                .expect("Chan not available in map")
                .push(DataLine::new(lost, tag, edge, sweep, time));
        }

    Ok(map_of_data)
}

/// Parse binary list files generated by a multiscaler.
/// Parameters:
/// fname - str
pub fn analyze_lst(fname: &str, start_of_data: usize, range: u64,
                   timepatch: &str, channel_map: Vec<u8>)
    -> Result<HashMap<u8, Vec<DataLine>>, Error> {

    let data_with_headers = FileBuffer::open(fname).expect("bad file name");
    let data_size: usize = (fs::metadata(fname)?.len() - start_of_data as u64) as usize;
    let data = &data_with_headers[start_of_data..];
    // Open the file and convert it to a usable format

    let chan_map = create_channel_map(data_size, channel_map);
    let tp_enum = Timepatch::new(timepatch);
    let processed_data = match tp_enum {
        Timepatch::Tp0(func) => func(data, range, &TimepatchBits::new(timepatch),chan_map),
        Timepatch::Tp5(func) => func(data, range, &TimepatchBits::new(timepatch),chan_map),
        Timepatch::Tp1(func) => func(data, range, &TimepatchBits::new(timepatch),chan_map),
        Timepatch::Tp1a(func) => func(data, range,&TimepatchBits::new(timepatch), chan_map),
        Timepatch::Tp2a(func) => func(data, range,&TimepatchBits::new(timepatch), chan_map),
        Timepatch::Tp22(func) => func(data, range,&TimepatchBits::new(timepatch), chan_map),
        Timepatch::Tp32(func) => func(data, range,&TimepatchBits::new(timepatch), chan_map),
        Timepatch::Tp2(func) => func(data, range, &TimepatchBits::new(timepatch),chan_map),
        Timepatch::Tp5b(func) => func(data, range,&TimepatchBits::new(timepatch), chan_map),
        Timepatch::TpDb(func) => func(data, range,&TimepatchBits::new(timepatch), chan_map),
        Timepatch::Tpf3(func) => func(data, range,&TimepatchBits::new(timepatch), chan_map),
        Timepatch::Tp43(func) => func(data, range,&TimepatchBits::new(timepatch), chan_map),
        Timepatch::Tpc3(func) => func(data, range,&TimepatchBits::new(timepatch), chan_map),
        Timepatch::Tp3(func) => func(data, range, &TimepatchBits::new(timepatch),chan_map),
    };
    processed_data

}