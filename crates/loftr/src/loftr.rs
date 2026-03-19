#[cfg(test)]
use tch::IValue;
use tch::{Device, Kind, Tensor};

use crate::error::LoftrError;

#[derive(Debug)]
pub struct LoftrMatches {
    pub keypoints0: Tensor,
    pub keypoints1: Tensor,
    pub confidence: Tensor,
    pub batch_indexes: Tensor,
}

/// Converts supported `LoFTR` image layouts into normalized `[B, 1, H, W]` tensors.
///
/// # Errors
///
/// Returns [`LoftrError::InvalidInput`] when `image` is not one of the supported
/// grayscale or RGB shapes accepted by the model.
pub fn normalize_loftr_image(image: &Tensor, device: Device) -> Result<Tensor, LoftrError> {
    let image = image.f_to_device(device)?.f_to_kind(Kind::Float)?;
    let dims = image.size();
    let image = match dims.as_slice() {
        [h, w] if *h > 0 && *w > 0 => image.unsqueeze(0).unsqueeze(0),
        [1, h, w] if *h > 0 && *w > 0 => image.unsqueeze(0),
        [3, h, w] if *h > 0 && *w > 0 => image
            .mean_dim([0].as_slice(), true, Kind::Float)
            .unsqueeze(0),
        [b, 1, h, w] if *b > 0 && *h > 0 && *w > 0 => image,
        [b, 3, h, w] if *b > 0 && *h > 0 && *w > 0 => {
            image.mean_dim([1].as_slice(), true, Kind::Float)
        }
        _ => {
            return Err(LoftrError::InvalidInput(format!(
                "LoFTR expects [H,W], [1,H,W], [3,H,W], [B,1,H,W], or [B,3,H,W]; got {dims:?}"
            )));
        }
    };

    let image = if image.max().double_value(&[]) > 1.0 {
        image / 255.0
    } else {
        image
    };

    Ok(image.clamp(0.0, 1.0))
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
