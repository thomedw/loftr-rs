use std::{env, path::Path};

use image::{DynamicImage, imageops::FilterType};
use loftr::{LoftrConfig, LoftrModel};
use tch::{Device, Kind, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: cargo run -p loftr --example match_pair -- <weights> <left> <right>");
        std::process::exit(2);
    }

    let mut model = LoftrModel::new(Device::Cpu, LoftrConfig::outdoor())?;
    model.load_weights(&args[1])?;

    let left = load_grayscale(Path::new(&args[2]))?;
    let right = load_grayscale(Path::new(&args[3]))?;
    let matches = model.forward(&left, &right)?;

    println!("matches={}", matches.confidence.size()[0]);
    Ok(())
}

fn load_grayscale(path: &Path) -> Result<Tensor, Box<dyn std::error::Error>> {
    let image = image::open(path)?;
    let image = resize_for_loftr(&image);
    let image = image.to_luma32f();
    let height = i64::from(image.height());
    let width = i64::from(image.width());
    let data = image.into_raw();
    let tensor = Tensor::from_slice(&data)
        .view([1, height, width])
        .unsqueeze(0)
        .to_kind(Kind::Float);
    Ok(tensor)
}

fn resize_for_loftr(image: &DynamicImage) -> DynamicImage {
    image.resize_exact(960, 540, FilterType::Triangle)
}
