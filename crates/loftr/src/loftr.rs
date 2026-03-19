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
mod tests;
