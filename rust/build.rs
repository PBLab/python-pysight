extern crate cbindgen;

fn main() {
    let crate_dir = r"C:\Users\Hagai\Documents\GitHub\pysight\rust";
    let mut config: cbindgen::Config = Default::default();
    config.language = cbindgen::Language::C;
    cbindgen::generate_with_config(crate_dir, config)
      .unwrap()
      .write_to_file("target/pysight.h");
}