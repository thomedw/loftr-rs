use std::{
    error::Error,
    path::{Path, PathBuf},
};

use image::{DynamicImage, imageops::FilterType};
use loftr::{LoftrConfig, LoftrModel};
use tch::{Device, Kind, Tensor};

const FIXTURE_SETUP_HINT: &str =
    "run `./scripts/prepare_test_fixtures.sh` from the workspace root and retry";

#[test]
fn end_to_end_matching_with_prepared_fixtures() -> Result<(), Box<dyn Error>> {
    let weights = fixture_path("loftr_outdoor_state_dict.safetensors")?;
    let left = fixture_path("kn_church-2.jpg")?;
    let right = fixture_path("kn_church-8.jpg")?;

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

fn fixture_path(name: &str) -> Result<PathBuf, Box<dyn Error>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    if path.is_file() {
        Ok(path)
    } else {
        Err(format!(
            "missing test fixture `{name}` at `{}`; {FIXTURE_SETUP_HINT}",
            path.display()
        )
        .into())
    }
}

fn load_grayscale(path: &Path) -> Result<Tensor, image::ImageError> {
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
