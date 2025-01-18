use std::{fs::File, io::Read};
use toml::Table;

/// Parses config.toml into toml 's 'Table'
pub fn parse_config() -> Table {
    let mut toml = String::new();
    File::open("config.toml")
        .expect("unable to read config")
        .read_to_string(&mut toml)
        .unwrap();
    toml.parse::<Table>().expect("failed to parse config")
}
