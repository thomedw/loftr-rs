#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::{env, path::PathBuf};

use image::{DynamicImage, imageops::FilterType};
use loftr::{LoftrConfig, LoftrModel};
use tch::{Device, Kind, Tensor};

#[test]
#[ignore = "requires local weights and image paths via environment variables"]
fn end_to_end_matching_with_local_weights() {
    let weights = env_path("LOFTR_TEST_WEIGHTS");
    let left = env_path("LOFTR_TEST_LEFT");
    let right = env_path("LOFTR_TEST_RIGHT");

    let mut model = LoftrModel::new(Device::Cpu, LoftrConfig::outdoor()).expect("model");
    model.load_weights(&weights).expect("weights");

    let left = load_grayscale(&left).expect("left image");
    let right = load_grayscale(&right).expect("right image");
    let matches = model.forward(&left, &right).expect("forward");

    assert_eq!(matches.keypoints0.size()[1], 2);
    assert_eq!(matches.keypoints1.size()[1], 2);
    assert_eq!(matches.confidence.size()[0], matches.keypoints0.size()[0]);
}

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(env::var(name).unwrap_or_else(|_| panic!("missing environment variable {name}")))
}

fn load_grayscale(path: &PathBuf) -> Result<Tensor, Box<dyn std::error::Error>> {
    let image = image::open(path)?;
    let image = resize_for_loftr(image);
    let image = image.to_luma32f();
    let height = i64::from(image.height());
    let width = i64::from(image.width());
    let data = image.into_raw();
    Ok(Tensor::from_slice(&data)
        .view([1, height, width])
        .unsqueeze(0)
        .to_kind(Kind::Float))
}

fn resize_for_loftr(image: DynamicImage) -> DynamicImage {
    image.resize_exact(960, 540, FilterType::Triangle)
}
