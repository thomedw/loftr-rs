use super::*;
use tch::IValue;

fn extract_loftr_matches(ivalue: IValue) -> Result<LoftrMatches, LoftrError> {
    let values = match ivalue {
        IValue::Tuple(values) | IValue::GenericList(values) => values,
        other => {
            return Err(LoftrError::InvalidConfig(format!(
                "LoFTR match extraction expected a tuple-like output, got {other:?}"
            )));
        }
    };
    if values.len() != 4 {
        return Err(LoftrError::InvalidConfig(format!(
            "LoFTR match extraction returned {} values; expected 4",
            values.len()
        )));
    }
    let mut values = values.into_iter();
    Ok(LoftrMatches {
        keypoints0: extract_tensor(values.next(), "keypoints0")?,
        keypoints1: extract_tensor(values.next(), "keypoints1")?,
        confidence: extract_tensor(values.next(), "confidence")?,
        batch_indexes: extract_tensor(values.next(), "batch_indexes")?,
    })
}

fn extract_tensor(value: Option<IValue>, label: &str) -> Result<Tensor, LoftrError> {
    match value {
        Some(IValue::Tensor(tensor)) => Ok(tensor),
        Some(other) => Err(LoftrError::InvalidConfig(format!(
            "LoFTR `{label}` output had unsupported type: {other:?}"
        ))),
        None => Err(LoftrError::InvalidConfig(format!(
            "LoFTR omitted `{label}` output"
        ))),
    }
}

#[test]
fn normalize_loftr_image_accepts_rgb_chw() -> Result<(), LoftrError> {
    let image = Tensor::from_slice(&[
        0.0_f32, 0.5, 1.0, 0.25, 0.25, 0.75, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
    ])
    .view([3, 2, 2]);
    let out = normalize_loftr_image(&image, Device::Cpu)?;
    assert_eq!(out.size(), vec![1, 1, 2, 2]);
    let min = out.min().double_value(&[]);
    let max = out.max().double_value(&[]);
    assert!(min >= 0.0);
    assert!(max <= 1.0);
    Ok(())
}

#[test]
fn normalize_loftr_image_rejects_invalid_rank() {
    let image = Tensor::zeros([2, 2, 2, 2, 2], (Kind::Float, Device::Cpu));
    match normalize_loftr_image(&image, Device::Cpu) {
        Ok(_) => panic!("invalid rank should fail"),
        Err(err) => assert!(format!("{err}").contains("LoFTR expects")),
    }
}

#[test]
fn normalize_loftr_image_scales_byte_range_to_unit_interval() -> Result<(), LoftrError> {
    let image = Tensor::from_slice(&[0.0_f32, 64.0, 128.0, 255.0]).view([1, 2, 2]);
    let out = normalize_loftr_image(&image, Device::Cpu)?;
    let values = out.reshape([-1]);
    assert!(values.double_value(&[0]) <= 1e-9);
    assert!((values.double_value(&[1]) - (64.0 / 255.0)).abs() < 1e-6);
    assert!((values.double_value(&[2]) - (128.0 / 255.0)).abs() < 1e-6);
    assert!((values.double_value(&[3]) - 1.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn extract_loftr_matches_parses_tuple_outputs() -> Result<(), LoftrError> {
    let keypoints0 = Tensor::zeros([4, 2], (Kind::Float, Device::Cpu));
    let keypoints1 = Tensor::ones([4, 2], (Kind::Float, Device::Cpu));
    let confidence = Tensor::full([4], 0.75, (Kind::Float, Device::Cpu));
    let batch_indexes = Tensor::zeros([4], (Kind::Int64, Device::Cpu));
    let out = extract_loftr_matches(IValue::Tuple(vec![
        IValue::Tensor(keypoints0.shallow_clone()),
        IValue::Tensor(keypoints1.shallow_clone()),
        IValue::Tensor(confidence.shallow_clone()),
        IValue::Tensor(batch_indexes.shallow_clone()),
    ]))?;
    assert_eq!(out.keypoints0.size(), vec![4, 2]);
    assert_eq!(out.keypoints1.size(), vec![4, 2]);
    assert_eq!(out.confidence.size(), vec![4]);
    assert_eq!(out.batch_indexes.kind(), Kind::Int64);
    Ok(())
}

#[test]
fn extract_loftr_matches_rejects_wrong_arity() {
    match extract_loftr_matches(IValue::Tuple(vec![IValue::Tensor(Tensor::zeros(
        [1],
        (Kind::Float, Device::Cpu),
    ))])) {
        Ok(_) => panic!("wrong arity should fail"),
        Err(err) => assert!(format!("{err}").contains("expected 4")),
    }
}
