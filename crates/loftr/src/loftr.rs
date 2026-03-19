use tch::{Device, Kind, Tensor};

use crate::error::LoftrError;

/// Matched keypoints and confidences returned by `LoFTR` inference.
///
/// Each tensor is aligned by row index: row `i` in `keypoints0`,
/// `keypoints1`, `confidence`, and `batch_indexes` describes one match.
#[derive(Debug)]
pub struct LoftrMatches {
    /// Matched points from the left image as `[M, 2]` `(x, y)` coordinates.
    pub keypoints0: Tensor,
    /// Matched points from the right image as `[M, 2]` `(x, y)` coordinates.
    pub keypoints1: Tensor,
    /// Match confidence scores as a rank-1 tensor with length `M`.
    pub confidence: Tensor,
    /// Batch indexes for each match as a rank-1 tensor with length `M`.
    pub batch_indexes: Tensor,
}

/// Converts supported `LoFTR` image layouts into normalized `[B, 1, H, W]` tensors.
///
/// Accepted inputs are grayscale or RGB tensors in `[H, W]`, `[1, H, W]`,
/// `[3, H, W]`, `[B, 1, H, W]`, or `[B, 3, H, W]` layout. RGB inputs are
/// reduced to grayscale by channel averaging. Inputs with values above `1.0`
/// are treated as byte-range images and rescaled to `[0, 1]`.
///
/// # Arguments
///
/// * `image` - Input tensor in one of the supported grayscale or RGB layouts.
/// * `device` - Target device for the normalized output tensor.
///
/// # Returns
///
/// A normalized grayscale tensor with shape `[B, 1, H, W]`.
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
