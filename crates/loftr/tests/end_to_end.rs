use std::{env, path::PathBuf};

use image::{DynamicImage, imageops::FilterType};
use loftr::{LoftrConfig, LoftrModel};
use tch::{Device, Kind, Tensor};

#[test]
#[ignore = "requires local weights and image paths via environment variables"]
fn end_to_end_matching_with_local_weights() -> Result<(), Box<dyn std::error::Error>> {
    let weights = env_path("LOFTR_TEST_WEIGHTS")?;
    let left = env_path("LOFTR_TEST_LEFT")?;
    let right = env_path("LOFTR_TEST_RIGHT")?;

    let mut model = LoftrModel::new(Device::Cpu, LoftrConfig::outdoor())?;
    model.load_weights(&weights)?;

    let left = load_grayscale(&left)?;
    let right = load_grayscale(&right)?;
    let matches = model.forward(&left, &right)?;

    assert_eq!(matches.keypoints0.size()[1], 2);
    assert_eq!(matches.keypoints1.size()[1], 2);
    assert_eq!(matches.confidence.size()[0], matches.keypoints0.size()[0]);
    Ok(())
}

fn env_path(name: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    match env::var(name) {
        Ok(path) => Ok(PathBuf::from(path)),
        Err(_) => Err(format!("missing environment variable {name}").into()),
    }
}

fn load_grayscale(path: &PathBuf) -> Result<Tensor, image::ImageError> {
    let image = image::open(path)?;
    let image = resize_for_loftr(&image);
    let image = image.to_luma32f();
    let height = i64::from(image.height());
    let width = i64::from(image.width());
    let data = image.into_raw();
    Ok(Tensor::from_slice(&data)
        .view([1, height, width])
        .unsqueeze(0)
        .to_kind(Kind::Float))
}

fn resize_for_loftr(image: &DynamicImage) -> DynamicImage {
    image.resize_exact(960, 540, FilterType::Triangle)
}
