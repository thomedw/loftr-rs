use super::*;
use tch::Device;

#[test]
fn empty_matches_return_coarse_points() -> Result<(), LoftrError> {
    let data = FineMatchingData {
        hw0_i: (16, 16),
        hw0_f: (8, 8),
        mkpts0_c: Tensor::zeros([0, 2], (Kind::Float, Device::Cpu)),
        mkpts1_c: Tensor::zeros([0, 2], (Kind::Float, Device::Cpu)),
        mconf: Tensor::zeros([0], (Kind::Float, Device::Cpu)),
        b_ids: Tensor::zeros([0], (Kind::Int64, Device::Cpu)),
        scale1: None,
    };
    let out = FineMatching::forward(
        &Tensor::zeros([0, 9, 4], (Kind::Float, Device::Cpu)),
        &Tensor::zeros([0, 9, 4], (Kind::Float, Device::Cpu)),
        &data,
    )?;
    assert_eq!(out.mkpts0_f.size(), vec![0, 2]);
    assert_eq!(out.mkpts1_f.size(), vec![0, 2]);
    Ok(())
}

#[test]
fn forward_produces_offset_for_peaked_heatmap() -> Result<(), LoftrError> {
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

    let out = FineMatching::forward(&feat_f0, &feat_f1, &data)?;
    assert_eq!(out.mkpts0_f.size(), vec![1, 2]);
    assert_eq!(out.mkpts1_f.size(), vec![1, 2]);
    assert!(out.mkpts1_f.double_value(&[0, 0]) > 101.5);
    assert!(out.mkpts1_f.double_value(&[0, 1]) > 201.5);
    Ok(())
}

#[test]
fn forward_uses_batch_scale1_when_present() -> Result<(), LoftrError> {
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

    let out = FineMatching::forward(&feat_f0, &feat_f1, &data)?;
    let first_x = out.mkpts1_f.double_value(&[0, 0]);
    let second_x = out.mkpts1_f.double_value(&[1, 0]);
    assert!(second_x > first_x + 1.0);
    Ok(())
}
