mod support;

use std::error::Error;

use loftr::LoftrConfig;
use support::{ReferenceFixture, load_model};
use tch::{Kind, Tensor};

#[test]
fn outdoor_matches_kornia_reference_fixture() -> Result<(), Box<dyn Error>> {
    let fixture = ReferenceFixture::load("loftr_outdoor_reference")?;
    let mut model = load_model(
        "loftr_outdoor_state_dict.safetensors",
        LoftrConfig::outdoor(),
    )?;

    let image0 = fixture.tensor("image0")?;
    let image1 = fixture.tensor("image1")?;
    let matches = model.forward(&image0, &image1)?;
    let expected0 = fixture.tensor("loftr_outdoor_tentatives0")?;
    let expected1 = fixture.tensor("loftr_outdoor_tentatives1")?;

    assert_eq!(matches.keypoints0.size(), expected0.size());
    assert_eq!(matches.keypoints1.size(), expected1.size());
    assert_eq!(matches.confidence.size()[0], expected0.size()[0]);
    assert_tensors_close(
        "outdoor keypoints0",
        &matches.keypoints0,
        &expected0,
        1e-4,
        1e-3,
    )?;
    assert_tensors_close(
        "outdoor keypoints1",
        &matches.keypoints1,
        &expected1,
        1e-4,
        1e-3,
    )?;
    Ok(())
}

#[test]
fn indoor_matches_kornia_reference_fixture() -> Result<(), Box<dyn Error>> {
    let fixture = ReferenceFixture::load("loftr_indoor_reference")?;
    let mut model = load_model("loftr_indoor_state_dict.safetensors", LoftrConfig::indoor())?;

    let image0 = fixture.tensor("image0")?;
    let image1 = fixture.tensor("image1")?;
    let matches = model.forward(&image0, &image1)?;
    let expected0 = fixture.tensor("loftr_indoor_tentatives0")?;
    let expected1 = fixture.tensor("loftr_indoor_tentatives1")?;

    assert_eq!(matches.keypoints0.size(), expected0.size());
    assert_eq!(matches.keypoints1.size(), expected1.size());
    assert_eq!(matches.confidence.size()[0], expected0.size()[0]);
    assert_tensors_close(
        "indoor keypoints0",
        &matches.keypoints0,
        &expected0,
        1e-4,
        1e-3,
    )?;
    assert_tensors_close(
        "indoor keypoints1",
        &matches.keypoints1,
        &expected1,
        1e-4,
        1e-3,
    )?;
    Ok(())
}

fn assert_tensors_close(
    label: &str,
    actual: &Tensor,
    expected: &Tensor,
    rtol: f64,
    atol: f64,
) -> Result<(), Box<dyn Error>> {
    let actual = actual.f_to_kind(Kind::Double)?;
    let expected = expected.f_to_kind(Kind::Double)?;
    if actual.size() != expected.size() {
        return Err(format!(
            "{label} size mismatch: actual={:?}, expected={:?}",
            actual.size(),
            expected.size()
        )
        .into());
    }
    if actual.allclose(&expected, rtol, atol, false) {
        Ok(())
    } else {
        let max_abs_diff = (&actual - &expected).abs().max().double_value(&[]);
        Err(
            format!("{label} mismatch: max_abs_diff={max_abs_diff:.6}, rtol={rtol}, atol={atol}")
                .into(),
        )
    }
}
