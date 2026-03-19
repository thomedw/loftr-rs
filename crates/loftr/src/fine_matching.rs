use tch::{Kind, Tensor};

use crate::error::LoftrError;

#[derive(Debug)]
pub struct FineMatchingData {
    pub hw0_i: (i64, i64),
    pub hw0_f: (i64, i64),
    pub mkpts0_c: Tensor,
    pub mkpts1_c: Tensor,
    pub mconf: Tensor,
    pub b_ids: Tensor,
    pub scale1: Option<Tensor>,
}

#[derive(Debug)]
pub struct FineMatchingOutput {
    pub mkpts0_f: Tensor,
    pub mkpts1_f: Tensor,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct FineMatching;

impl FineMatching {
    pub fn forward(
        &self,
        feat_f0: &Tensor,
        feat_f1: &Tensor,
        data: &FineMatchingData,
    ) -> Result<FineMatchingOutput, LoftrError> {
        validate_fine_match_inputs(feat_f0, feat_f1, data)?;

        let dims = feat_f0.size();
        let match_count = dims[0];
        let window_area = dims[1];
        let channel_dim = dims[2];
        let window_size = perfect_square_side(window_area)?;
        if match_count == 0 {
            return Ok(FineMatchingOutput {
                mkpts0_f: data.mkpts0_c.shallow_clone(),
                mkpts1_f: data.mkpts1_c.shallow_clone(),
            });
        }

        let feat_f0_center = feat_f0.select(1, window_area / 2);
        let sim_matrix = Tensor::einsum("mc,mrc->mr", &[&feat_f0_center, feat_f1], None::<&[i64]>);
        let softmax_temp = 1.0 / (channel_dim as f64).sqrt();
        let heatmap = (sim_matrix * softmax_temp).softmax(1, Kind::Float).view([
            match_count,
            window_size,
            window_size,
        ]);

        let grid = normalized_meshgrid(window_size, heatmap.device(), heatmap.kind());
        let heatmap_flat = heatmap.view([match_count, window_area, 1]);
        let coords_normalized =
            (&heatmap_flat * &grid).sum_dim_intlist([1].as_slice(), false, Kind::Float);
        let variance =
            (&heatmap_flat * grid.square()).sum_dim_intlist([1].as_slice(), false, Kind::Float)
                - coords_normalized.square();
        let _ = variance
            .clamp_min(1e-10)
            .sqrt()
            .sum_dim_intlist([1].as_slice(), true, Kind::Float);

        let mkpts0_f = data.mkpts0_c.shallow_clone();
        let scale1 = scale_factor_for_matches(data, feat_f0.device())?;
        let mkpts1_f = &data.mkpts1_c + coords_normalized * ((window_size / 2) as f64) * scale1;

        Ok(FineMatchingOutput { mkpts0_f, mkpts1_f })
    }
}

fn validate_fine_match_inputs(
    feat_f0: &Tensor,
    feat_f1: &Tensor,
    data: &FineMatchingData,
) -> Result<(), LoftrError> {
    let left_dims = feat_f0.size();
    let right_dims = feat_f1.size();
    if left_dims.len() != 3 || right_dims.len() != 3 {
        return Err(LoftrError::InvalidConfig(format!(
            "FineMatching expects [M,WW,C] tensors; got feat_f0={:?}, feat_f1={:?}",
            left_dims, right_dims
        )));
    }
    if left_dims != right_dims {
        return Err(LoftrError::InvalidConfig(format!(
            "FineMatching requires matching tensor shapes; got feat_f0={:?}, feat_f1={:?}",
            left_dims, right_dims
        )));
    }
    if data.hw0_i.0 <= 0 || data.hw0_i.1 <= 0 || data.hw0_f.0 <= 0 || data.hw0_f.1 <= 0 {
        return Err(LoftrError::InvalidConfig(format!(
            "FineMatching requires positive image/fine shapes; got hw0_i={:?}, hw0_f={:?}",
            data.hw0_i, data.hw0_f
        )));
    }
    if data.hw0_i.0 % data.hw0_f.0 != 0 {
        return Err(LoftrError::InvalidConfig(format!(
            "FineMatching requires integer scale between hw0_i and hw0_f; got hw0_i={:?}, hw0_f={:?}",
            data.hw0_i, data.hw0_f
        )));
    }

    let expected = left_dims[0];
    for (label, tensor, expected_last) in [
        ("mkpts0_c", &data.mkpts0_c, Some(2)),
        ("mkpts1_c", &data.mkpts1_c, Some(2)),
        ("mconf", &data.mconf, None),
        ("b_ids", &data.b_ids, None),
    ] {
        let dims = tensor.size();
        if dims.first().copied().unwrap_or_default() != expected {
            return Err(LoftrError::InvalidConfig(format!(
                "FineMatching `{label}` length mismatch: expected {expected}, got {:?}",
                dims
            )));
        }
        if let Some(last) = expected_last {
            if dims.len() != 2 || dims[1] != last {
                return Err(LoftrError::InvalidConfig(format!(
                    "FineMatching `{label}` expects [M,{last}]; got {:?}",
                    dims
                )));
            }
        } else if dims.len() != 1 {
            return Err(LoftrError::InvalidConfig(format!(
                "FineMatching `{label}` expects [M]; got {:?}",
                dims
            )));
        }
    }
    Ok(())
}

fn perfect_square_side(window_area: i64) -> Result<i64, LoftrError> {
    if window_area <= 0 {
        return Err(LoftrError::InvalidConfig(format!(
            "FineMatching requires positive window area; got {window_area}"
        )));
    }
    let window_size = (window_area as f64).sqrt() as i64;
    if window_size * window_size != window_area {
        return Err(LoftrError::InvalidConfig(format!(
            "FineMatching window area must be a perfect square; got {window_area}"
        )));
    }
    Ok(window_size)
}

fn normalized_meshgrid(window_size: i64, device: tch::Device, kind: Kind) -> Tensor {
    let coords = if window_size == 1 {
        Tensor::zeros([1], (kind, device))
    } else {
        Tensor::linspace(-1.0, 1.0, window_size, (kind, device))
    };
    let ys = coords.unsqueeze(1).repeat([1, window_size]);
    let xs = coords.unsqueeze(0).repeat([window_size, 1]);
    Tensor::stack(&[xs.reshape([-1]), ys.reshape([-1])], 1).unsqueeze(0)
}

fn scale_factor_for_matches(
    data: &FineMatchingData,
    device: tch::Device,
) -> Result<Tensor, LoftrError> {
    let scale = data.hw0_i.0 as f64 / data.hw0_f.0 as f64;
    match &data.scale1 {
        Some(scale1) => {
            let dims = scale1.size();
            if dims.len() != 1 {
                return Err(LoftrError::InvalidConfig(format!(
                    "FineMatching scale1 expects [B]; got {:?}",
                    dims
                )));
            }
            let b_ids = data.b_ids.f_to_device(device)?.f_to_kind(Kind::Int64)?;
            Ok(scale1
                .f_to_device(device)?
                .f_to_kind(Kind::Float)?
                .index_select(0, &b_ids)
                .unsqueeze(1)
                * scale)
        }
        None => Ok(Tensor::from(scale as f32).to_device(device).reshape([1, 1])),
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn empty_matches_return_coarse_points() {
        let data = FineMatchingData {
            hw0_i: (16, 16),
            hw0_f: (8, 8),
            mkpts0_c: Tensor::zeros([0, 2], (Kind::Float, Device::Cpu)),
            mkpts1_c: Tensor::zeros([0, 2], (Kind::Float, Device::Cpu)),
            mconf: Tensor::zeros([0], (Kind::Float, Device::Cpu)),
            b_ids: Tensor::zeros([0], (Kind::Int64, Device::Cpu)),
            scale1: None,
        };
        let out = FineMatching
            .forward(
                &Tensor::zeros([0, 9, 4], (Kind::Float, Device::Cpu)),
                &Tensor::zeros([0, 9, 4], (Kind::Float, Device::Cpu)),
                &data,
            )
            .expect("fine matching");
        assert_eq!(out.mkpts0_f.size(), vec![0, 2]);
        assert_eq!(out.mkpts1_f.size(), vec![0, 2]);
    }

    #[test]
    fn forward_produces_offset_for_peaked_heatmap() {
        let feat_f0 =
            Tensor::from_slice(&[0.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).view([1, 9, 1]);
        let feat_f1 =
            Tensor::from_slice(&[0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0]).view([1, 9, 1]);
        let data = FineMatchingData {
            hw0_i: (16, 16),
            hw0_f: (8, 8),
            mkpts0_c: Tensor::from_slice(&[10.0_f32, 20.0]).view([1, 2]),
            mkpts1_c: Tensor::from_slice(&[100.0_f32, 200.0]).view([1, 2]),
            mconf: Tensor::from_slice(&[0.9_f32]),
            b_ids: Tensor::from_slice(&[0_i64]),
            scale1: None,
        };

        let out = FineMatching
            .forward(&feat_f0, &feat_f1, &data)
            .expect("fine matching");
        assert_eq!(out.mkpts0_f.size(), vec![1, 2]);
        assert_eq!(out.mkpts1_f.size(), vec![1, 2]);
        assert!(out.mkpts1_f.double_value(&[0, 0]) > 101.5);
        assert!(out.mkpts1_f.double_value(&[0, 1]) > 201.5);
    }

    #[test]
    fn forward_uses_batch_scale1_when_present() {
        let feat_f0 = Tensor::ones([2, 9, 1], (Kind::Float, Device::Cpu));
        let mut right_values = vec![0.0_f32; 18];
        right_values[2] = 8.0;
        right_values[11] = 8.0;
        let feat_f1 = Tensor::from_slice(&right_values).view([2, 9, 1]);
        let data = FineMatchingData {
            hw0_i: (16, 16),
            hw0_f: (8, 8),
            mkpts0_c: Tensor::zeros([2, 2], (Kind::Float, Device::Cpu)),
            mkpts1_c: Tensor::zeros([2, 2], (Kind::Float, Device::Cpu)),
            mconf: Tensor::ones([2], (Kind::Float, Device::Cpu)),
            b_ids: Tensor::from_slice(&[0_i64, 1]),
            scale1: Some(Tensor::from_slice(&[1.0_f32, 2.0])),
        };

        let out = FineMatching
            .forward(&feat_f0, &feat_f1, &data)
            .expect("fine matching");
        let first_x = out.mkpts1_f.double_value(&[0, 0]);
        let second_x = out.mkpts1_f.double_value(&[1, 0]);
        assert!(second_x > first_x + 1.0);
    }
}
