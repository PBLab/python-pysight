use std::sync::Mutex;

use crate::parsing::*;

/// Populates a vector of mutex-controlled vectors with the valid active channels of the
/// experiment. Each channel is a vector, which contains another vector, holding four
/// other vectors - each of them holding a parsed value - either "lost", "tag", "edge" 
/// or "time".The maximal size of each of these vectors is `data_size + 1` if there was
/// any data in that channel. Otherwise, its size is 0. Each channel vector is also wrapped in a
/// mutex to allow for multi-threaded parsing.
/// Note - I don't use an Option<Mutex<Vec>>> here since I wasn't able to make it compile,
/// although it probably is the more ergonomic version.
pub fn create_channel_vec_u8(timepatch: &str, active_channels: Vec<u8>,
                             data_size: usize) -> Vec<Mutex<DataLineU8>> {
    let chan_with_data = generate_data_vectors_u8(data_size, timepatch);
    let empty_chan = DataLineU8::new(vec![], vec![], vec![], vec![]);
    let mut chans = Vec::with_capacity(NUM_OF_INPUT_CHANNELS);
    for is_active in active_channels.iter() {
        if is_active == &1u8 {
            chans.push(Mutex::new(chan_with_data.clone()));
        } else {
            chans.push(Mutex::new(empty_chan.clone()));
        };
    };
    chans
}

pub fn create_channel_vec_u16(timepatch: &str, active_channels: Vec<u8>,
                              data_size: usize) -> Vec<Mutex<DataLineU16>> {
    let chan_with_data = generate_data_vectors_u16(data_size, timepatch);
    let empty_chan = DataLineU16::new(vec![], vec![], vec![], vec![]);
    let mut chans = Vec::with_capacity(NUM_OF_INPUT_CHANNELS);
    for is_active in active_channels.iter() {
        if is_active == &1u8 {
            chans.push(Mutex::new(chan_with_data.clone()));
        } else {
            chans.push(Mutex::new(empty_chan.clone()));
        };
    };
    chans
}


fn generate_data_vectors_u8(data_size: usize, timepatch: &str) -> DataLineU8 {
    let num_of_lines: usize = match timepatch {
        "2a" | "22" => calc_num_of_lines(data_size, 6),
        "3" => calc_num_of_lines(data_size, 8),
        _ => panic!("Timepatch not found for u8 variant {}", timepatch),
    };
    let ve: DataLineU8 = match timepatch {
        "2a" => DataLineU8::new(vec![], 
                                Vec::with_capacity(num_of_lines + 1), 
                                Vec::with_capacity(num_of_lines), 
                                Vec::with_capacity(data_size + 1)),
        "22" => DataLineU8::new(vec![], 
                                Vec::with_capacity(num_of_lines + 1), 
                                Vec::with_capacity(num_of_lines), 
                                Vec::with_capacity(data_size + 1)),
        "3" => DataLineU8::new(Vec::with_capacity(num_of_lines + 1), 
                               Vec::with_capacity(num_of_lines + 1), 
                               Vec::with_capacity(num_of_lines), 
                               Vec::with_capacity(data_size + 1)),
        _ => panic!("Invalid timepatch value: {}", timepatch),
    };
    ve
}

/// Each timepatch value correlates to a specific vector composition which is detailed here.
/// The order of vecs is lost, tag, edge, and time.
fn generate_data_vectors_u16(data_size: usize, timepatch: &str) -> DataLineU16 {
    let num_of_lines: usize = match timepatch {
        "0" => calc_num_of_lines(data_size, 2),
        "5" | "1" => calc_num_of_lines(data_size, 4),
        "1a" | "32" | "2" => calc_num_of_lines(data_size, 6),
        "5b" | "Db" | "f3" | "43" | "c3" => calc_num_of_lines(data_size, 8),
        _ => panic!("Timepatch not found for variant u16 {}", timepatch),
    };

    let ve: DataLineU16 = match timepatch {
        "0" => DataLineU16::new(vec![], vec![], Vec::with_capacity(num_of_lines + 1), Vec::with_capacity(data_size + 1)),
        "5" => DataLineU16::new(vec![], vec![], Vec::with_capacity(num_of_lines + 1), Vec::with_capacity(data_size + 1)), 
        "1" => DataLineU16::new(vec![], vec![], Vec::with_capacity(num_of_lines + 1), Vec::with_capacity(data_size + 1)),
        "1a" => DataLineU16::new(vec![], vec![], Vec::with_capacity(num_of_lines + 1), Vec::with_capacity(data_size + 1)),
        "32" => DataLineU16::new(Vec::with_capacity(num_of_lines + 1), vec![], Vec::with_capacity(num_of_lines), Vec::with_capacity(data_size + 1)),
        "2" => DataLineU16::new(vec![], vec![], Vec::with_capacity(num_of_lines + 1), Vec::with_capacity(data_size + 1)),
        "5b" => DataLineU16::new(Vec::with_capacity(num_of_lines + 1), Vec::with_capacity(num_of_lines * 2 + 1), Vec::with_capacity(num_of_lines), Vec::with_capacity(data_size + 1)),
        "Db" => DataLineU16::new(vec![], Vec::with_capacity(num_of_lines * 2 + 1), Vec::with_capacity(num_of_lines), Vec::with_capacity(data_size + 1)),
        "f3" => DataLineU16::new(Vec::with_capacity(num_of_lines + 1), Vec::with_capacity(num_of_lines * 2 + 1), Vec::with_capacity(num_of_lines), Vec::with_capacity(data_size + 1)),
        "43" => DataLineU16::new(Vec::with_capacity(num_of_lines + 1), Vec::with_capacity(num_of_lines * 2 + 1), Vec::with_capacity(num_of_lines), Vec::with_capacity(data_size + 1)),
        "c3" => DataLineU16::new(vec![], Vec::with_capacity(num_of_lines * 2 + 1), Vec::with_capacity(num_of_lines), Vec::with_capacity(data_size + 1)),
        _ => panic!("Invalid timepatch value for u16 variant: {}", timepatch),
    };
    ve
}

/// Find how many events were in the recording
fn calc_num_of_lines(data_size: usize, bytes: u8) -> usize {
    ((data_size / bytes as usize) + 1) as usize  
}


// pub fn create_channel_vec_seq(data_size: usize, active_channels: Vec<u8>) -> Vec<Vec<DataLine>> {
//     let vec_with_data = Vec::with_capacity(data_size + 1);
//     let vec_empty_chan = Vec::with_capacity(0);
//     let mut chans = Vec::with_capacity(NUM_OF_INPUT_CHANNELS);
//     for is_active in active_channels.iter() {
//         if is_active == &1u8 {
//             chans.push(vec_with_data.clone());
//         } else {
//             chans.push(vec_empty_chan.clone());
//         };
//     };
//     chans
// }


pub fn to_bits_u16(bitarray: &[u8; 4]) -> [u16; 4] {
    let time_bits = "1".repeat(bitarray[3] as usize); 
    let time_bits = u16::from_str_radix(&time_bits, 2).unwrap();

    let sweep_bits = "1".repeat(bitarray[2] as usize);
    let sweep_bits = u16::from_str_radix(&sweep_bits, 2).unwrap_or(0);

    let tag_bits = "1".repeat(bitarray[1] as usize);
    let tag_bits = u16::from_str_radix(&tag_bits, 2).unwrap_or(0);

    let lost_bit = "1".repeat(bitarray[0] as usize);
    let lost_bit = u16::from_str_radix(&lost_bit, 2).unwrap_or(0);

    let bitmap: [u16; 4] = [lost_bit, tag_bits, sweep_bits, time_bits];
    bitmap  
}

pub fn to_bits_u32(bitarray: &[u8; 4]) -> [u32; 4] {
    let time_bits = "1".repeat(bitarray[3] as usize); 
    let time_bits = u32::from_str_radix(&time_bits, 2).unwrap();

    let sweep_bits = "1".repeat(bitarray[2] as usize);
    let sweep_bits = u32::from_str_radix(&sweep_bits, 2).unwrap_or(0);

    let tag_bits = "1".repeat(bitarray[1] as usize);
    let tag_bits = u32::from_str_radix(&tag_bits, 2).unwrap_or(0);

    let lost_bit = "1".repeat(bitarray[0] as usize);
    let lost_bit = u32::from_str_radix(&lost_bit, 2).unwrap_or(0);

    let bitmap: [u32; 4] = [lost_bit, tag_bits, sweep_bits, time_bits];
    bitmap  
}

pub fn to_bits_u64(bitarray: &[u8; 4]) -> [u64; 4] {
    let time_bits = "1".repeat(bitarray[3] as usize); 
    let time_bits = u64::from_str_radix(&time_bits, 2).unwrap();

    let sweep_bits = "1".repeat(bitarray[2] as usize);
    let sweep_bits = u64::from_str_radix(&sweep_bits, 2).unwrap_or(0);

    let tag_bits = "1".repeat(bitarray[1] as usize);
    let tag_bits = u64::from_str_radix(&tag_bits, 2).unwrap_or(0);

    let lost_bit = "1".repeat(bitarray[0] as usize);
    let lost_bit = u64::from_str_radix(&lost_bit, 2).unwrap_or(0);

    let bitmap: [u64; 4] = [lost_bit, tag_bits, sweep_bits, time_bits];
    bitmap  
}