use tch::{Kind, Tensor};

use crate::{error::LoftrError, loftr_config::MatchCoarseConfig};

const INF: f64 = 1e9;

#[derive(Debug)]
pub struct CoarseMatchingData {
    pub hw0_i: (i64, i64),
    pub hw1_i: (i64, i64),
    pub hw0_c: (i64, i64),
    pub hw1_c: (i64, i64),
    pub scale0: Option<Tensor>,
    pub scale1: Option<Tensor>,
}

#[derive(Debug)]
pub struct CoarseMatchingOutput {
    pub conf_matrix: Tensor,
    pub b_ids: Tensor,
    pub i_ids: Tensor,
    pub j_ids: Tensor,
    pub m_bids: Tensor,
    pub mkpts0_c: Tensor,
    pub mkpts1_c: Tensor,
    pub mconf: Tensor,
}

#[derive(Debug, Clone)]
pub struct CoarseMatching {
    config: MatchCoarseConfig,
}

impl CoarseMatching {
    pub fn new(config: &MatchCoarseConfig) -> Result<Self, LoftrError> {
        if config.match_type != "dual_softmax" {
            return Err(LoftrError::InvalidConfig(format!(
                "CoarseMatching currently supports only dual_softmax; got `{}`",
                config.match_type
            )));
        }
        Ok(Self {
            config: config.clone(),
        })
    }

    pub fn forward(
        &self,
        feat_c0: &Tensor,
        feat_c1: &Tensor,
        data: &CoarseMatchingData,
        mask_c0: Option<&Tensor>,
        mask_c1: Option<&Tensor>,
    ) -> Result<CoarseMatchingOutput, LoftrError> {
        validate_coarse_matching_inputs(feat_c0, feat_c1, data, mask_c0, mask_c1)?;

        let feat_c0 = feat_c0 / (feat_c0.size()[2] as f64).sqrt();
        let feat_c1 = feat_c1 / (feat_c1.size()[2] as f64).sqrt();
        let mut sim_matrix = Tensor::einsum("nlc,nsc->nls", &[&feat_c0, &feat_c1], None::<&[i64]>)
            / self.config.dsmax_temperature;
        if let (Some(mask_c0), Some(mask_c1)) = (mask_c0, mask_c1) {
            let valid = mask_c0
                .f_to_device(sim_matrix.device())?
                .f_to_kind(Kind::Bool)?
                .unsqueeze(-1)
                .logical_and(
                    &mask_c1
                        .f_to_device(sim_matrix.device())?
                        .f_to_kind(Kind::Bool)?
                        .unsqueeze(1),
                );
            sim_matrix = sim_matrix.f_masked_fill(&valid.logical_not(), -INF)?;
        }
        let conf_matrix = sim_matrix.softmax(1, Kind::Float) * sim_matrix.softmax(2, Kind::Float);

        let threshold_mask =
            confidence_threshold_mask(&conf_matrix, data, self.config.thr, self.config.border_rm)?;
        let mutual_mask = conf_matrix
            .f_eq_tensor(&conf_matrix.max_dim(2, true).0)?
            .logical_and(&conf_matrix.f_eq_tensor(&conf_matrix.max_dim(1, true).0)?);
        let matches = threshold_mask.logical_and(&mutual_mask);
        let match_indices = matches.nonzero();

        let (b_ids, i_ids, j_ids, mconf) = if match_indices.size()[0] == 0 {
            empty_match_outputs(&conf_matrix)
        } else {
            let b_ids = match_indices.select(1, 0);
            let i_ids = match_indices.select(1, 1);
            let j_ids = match_indices.select(1, 2);
            let linear_ids = &b_ids * (conf_matrix.size()[1] * conf_matrix.size()[2])
                + &i_ids * conf_matrix.size()[2]
                + &j_ids;
            let mconf = conf_matrix.reshape([-1]).index_select(0, &linear_ids);
            (b_ids, i_ids, j_ids, mconf)
        };

        let pred_mask = mconf.ne(0.0);
        let m_bids = b_ids.masked_select(&pred_mask);
        let mkpts0_c = coarse_points(
            &i_ids,
            data.hw0_c.1,
            scaled_factors(data, &b_ids, true, conf_matrix.device())?,
        );
        let mkpts1_c = coarse_points(
            &j_ids,
            data.hw1_c.1,
            scaled_factors(data, &b_ids, false, conf_matrix.device())?,
        );

        Ok(CoarseMatchingOutput {
            conf_matrix,
            b_ids,
            i_ids,
            j_ids,
            m_bids,
            mkpts0_c: mkpts0_c
                .masked_select(&pred_mask.unsqueeze(1))
                .reshape([-1, 2])
                .to_kind(Kind::Float),
            mkpts1_c: mkpts1_c
                .masked_select(&pred_mask.unsqueeze(1))
                .reshape([-1, 2])
                .to_kind(Kind::Float),
            mconf: mconf.masked_select(&pred_mask),
        })
    }
}

fn validate_coarse_matching_inputs(
    feat_c0: &Tensor,
    feat_c1: &Tensor,
    data: &CoarseMatchingData,
    mask_c0: Option<&Tensor>,
    mask_c1: Option<&Tensor>,
) -> Result<(), LoftrError> {
    let left_dims = feat_c0.size();
    let right_dims = feat_c1.size();
    if left_dims.len() != 3 || right_dims.len() != 3 {
        return Err(LoftrError::InvalidConfig(format!(
            "CoarseMatching expects [N,L,C] tensors; got feat_c0={:?}, feat_c1={:?}",
            left_dims, right_dims
        )));
    }
    if left_dims[0] != right_dims[0] || left_dims[2] != right_dims[2] {
        return Err(LoftrError::InvalidConfig(format!(
            "CoarseMatching batch/channel mismatch: feat_c0={:?}, feat_c1={:?}",
            left_dims, right_dims
        )));
    }
    if left_dims[1] != data.hw0_c.0 * data.hw0_c.1 || right_dims[1] != data.hw1_c.0 * data.hw1_c.1 {
        return Err(LoftrError::InvalidConfig(format!(
            "CoarseMatching hw*_c mismatch: feat_c0={:?}, feat_c1={:?}, hw0_c={:?}, hw1_c={:?}",
            left_dims, right_dims, data.hw0_c, data.hw1_c
        )));
    }
    if let Some(mask) = mask_c0 {
        if mask.size() != vec![left_dims[0], left_dims[1]] {
            return Err(LoftrError::InvalidConfig(format!(
                "CoarseMatching mask_c0 expects [{}, {}]; got {:?}",
                left_dims[0],
                left_dims[1],
                mask.size()
            )));
        }
    }
    if let Some(mask) = mask_c1 {
        if mask.size() != vec![right_dims[0], right_dims[1]] {
            return Err(LoftrError::InvalidConfig(format!(
                "CoarseMatching mask_c1 expects [{}, {}]; got {:?}",
                right_dims[0],
                right_dims[1],
                mask.size()
            )));
        }
    }
    Ok(())
}

fn confidence_threshold_mask(
    conf_matrix: &Tensor,
    data: &CoarseMatchingData,
    threshold: f64,
    border_rm: i64,
) -> Result<Tensor, LoftrError> {
    let n = conf_matrix.size()[0];
    let h0 = data.hw0_c.0;
    let w0 = data.hw0_c.1;
    let h1 = data.hw1_c.0;
    let w1 = data.hw1_c.1;
    let threshold_mask = conf_matrix.gt(threshold);
    let border_mask = border_validity_mask(h0, w0, h1, w1, border_rm, conf_matrix.device());
    Ok(threshold_mask
        .reshape([n, h0, w0, h1, w1])
        .logical_and(&border_mask)
        .reshape([n, h0 * w0, h1 * w1]))
}

fn border_validity_mask(
    h0: i64,
    w0: i64,
    h1: i64,
    w1: i64,
    border_rm: i64,
    device: tch::Device,
) -> Tensor {
    if border_rm <= 0 {
        return Tensor::ones([1, h0, w0, h1, w1], (Kind::Bool, device));
    }
    let y0 = border_axis_mask(h0, border_rm, device).view([1, h0, 1, 1, 1]);
    let x0 = border_axis_mask(w0, border_rm, device).view([1, 1, w0, 1, 1]);
    let y1 = border_axis_mask(h1, border_rm, device).view([1, 1, 1, h1, 1]);
    let x1 = border_axis_mask(w1, border_rm, device).view([1, 1, 1, 1, w1]);
    y0.logical_and(&x0).logical_and(&y1).logical_and(&x1)
}

fn border_axis_mask(length: i64, border_rm: i64, device: tch::Device) -> Tensor {
    let indices = Tensor::arange(length, (Kind::Int64, device));
    indices
        .ge(border_rm)
        .logical_and(&indices.lt(length - border_rm))
}

fn empty_match_outputs(conf_matrix: &Tensor) -> (Tensor, Tensor, Tensor, Tensor) {
    let device = conf_matrix.device();
    (
        Tensor::zeros([0], (Kind::Int64, device)),
        Tensor::zeros([0], (Kind::Int64, device)),
        Tensor::zeros([0], (Kind::Int64, device)),
        Tensor::zeros([0], (Kind::Float, device)),
    )
}

fn scaled_factors(
    data: &CoarseMatchingData,
    b_ids: &Tensor,
    use_left: bool,
    device: tch::Device,
) -> Result<Tensor, LoftrError> {
    let scale = if use_left {
        data.hw0_i.0 as f64 / data.hw0_c.0 as f64
    } else {
        data.hw1_i.0 as f64 / data.hw1_c.0 as f64
    };
    let per_batch = if use_left { &data.scale0 } else { &data.scale1 };
    match per_batch {
        Some(per_batch) => Ok(per_batch
            .f_to_device(device)?
            .f_to_kind(Kind::Float)?
            .index_select(0, &b_ids.f_to_device(device)?.f_to_kind(Kind::Int64)?)
            .unsqueeze(1)
            * scale),
        None => Ok(Tensor::from(scale as f32).to_device(device).reshape([1, 1])),
    }
}

fn coarse_points(indices: &Tensor, width: i64, scale: Tensor) -> Tensor {
    let indices = indices.to_kind(Kind::Int64);
    let x = indices.remainder(width).to_kind(Kind::Float);
    let y = indices.floor_divide_scalar(width).to_kind(Kind::Float);
    Tensor::stack(&[x, y], 1) * scale
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::loftr_config::LoftrConfig;
    use tch::Device;

    #[test]
    fn dual_softmax_identity_features_match_diagonal() {
        let config = LoftrConfig::outdoor();
        let matcher = CoarseMatching::new(&MatchCoarseConfig {
            border_rm: 0,
            ..config.match_coarse
        })
        .expect("matcher");
        let feat_c0 = Tensor::eye(4, (Kind::Float, Device::Cpu)).view([1, 4, 4]);
        let feat_c1 = Tensor::eye(4, (Kind::Float, Device::Cpu)).view([1, 4, 4]);
        let out = matcher
            .forward(
                &feat_c0,
                &feat_c1,
                &CoarseMatchingData {
                    hw0_i: (8, 8),
                    hw1_i: (8, 8),
                    hw0_c: (2, 2),
                    hw1_c: (2, 2),
                    scale0: None,
                    scale1: None,
                },
                None,
                None,
            )
            .expect("forward");
        assert_eq!(out.b_ids.size(), vec![4]);
        assert_eq!(out.i_ids.size(), vec![4]);
        assert_eq!(out.j_ids.size(), vec![4]);
        assert_eq!(out.mconf.size(), vec![4]);
        assert!(out.mconf.min().double_value(&[]) > 0.2);
        assert_eq!(out.mkpts0_c.size(), vec![4, 2]);
        assert_eq!(out.mkpts1_c.size(), vec![4, 2]);
    }

    #[test]
    fn border_mask_removes_all_matches_on_tiny_grid() {
        let config = LoftrConfig::outdoor();
        let matcher = CoarseMatching::new(&config.match_coarse).expect("matcher");
        let feat_c0 = Tensor::eye(4, (Kind::Float, Device::Cpu)).view([1, 4, 4]);
        let feat_c1 = Tensor::eye(4, (Kind::Float, Device::Cpu)).view([1, 4, 4]);
        let out = matcher
            .forward(
                &feat_c0,
                &feat_c1,
                &CoarseMatchingData {
                    hw0_i: (8, 8),
                    hw1_i: (8, 8),
                    hw0_c: (2, 2),
                    hw1_c: (2, 2),
                    scale0: None,
                    scale1: None,
                },
                None,
                None,
            )
            .expect("forward");
        assert_eq!(out.b_ids.size(), vec![0]);
        assert_eq!(out.mconf.size(), vec![0]);
    }
}
