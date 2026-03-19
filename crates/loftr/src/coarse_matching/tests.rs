use super::*;
use crate::loftr_config::LoftrConfig;
use tch::Device;

#[test]
fn dual_softmax_identity_features_match_diagonal() -> Result<(), LoftrError> {
    let config = LoftrConfig::outdoor();
    let matcher = CoarseMatching::new(&MatchCoarseConfig {
        border_rm: 0,
        ..config.match_coarse
    })?;
    let feat_c0 = Tensor::eye(4, (Kind::Float, Device::Cpu)).view([1, 4, 4]);
    let feat_c1 = Tensor::eye(4, (Kind::Float, Device::Cpu)).view([1, 4, 4]);
    let out = matcher.forward(
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
    )?;
    assert_eq!(out.b_ids.size(), vec![4]);
    assert_eq!(out.i_ids.size(), vec![4]);
    assert_eq!(out.j_ids.size(), vec![4]);
    assert_eq!(out.mconf.size(), vec![4]);
    assert!(out.mconf.min().double_value(&[]) > 0.2);
    assert_eq!(out.mkpts0_c.size(), vec![4, 2]);
    assert_eq!(out.mkpts1_c.size(), vec![4, 2]);
    Ok(())
}

#[test]
fn border_mask_removes_all_matches_on_tiny_grid() -> Result<(), LoftrError> {
    let config = LoftrConfig::outdoor();
    let matcher = CoarseMatching::new(&config.match_coarse)?;
    let feat_c0 = Tensor::eye(4, (Kind::Float, Device::Cpu)).view([1, 4, 4]);
    let feat_c1 = Tensor::eye(4, (Kind::Float, Device::Cpu)).view([1, 4, 4]);
    let out = matcher.forward(
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
    )?;
    assert_eq!(out.b_ids.size(), vec![0]);
    assert_eq!(out.mconf.size(), vec![0]);
    Ok(())
}
