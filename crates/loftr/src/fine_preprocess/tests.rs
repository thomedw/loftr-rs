use super::*;
use tch::{Device, Kind, nn};

fn cpu_module(config: &LoftrConfig) -> Result<FinePreprocess, LoftrError> {
    let vs = nn::VarStore::new(Device::Cpu);
    FinePreprocess::new(&vs.root(), config)
}

#[test]
fn empty_matches_return_empty_windows() -> Result<(), LoftrError> {
    let config = LoftrConfig::outdoor();
    let module = cpu_module(&config)?;
    let data = FinePreprocessData {
        hw0_f: (4, 4),
        hw0_c: (2, 2),
        b_ids: Tensor::zeros([0], (Kind::Int64, Device::Cpu)),
        i_ids: Tensor::zeros([0], (Kind::Int64, Device::Cpu)),
        j_ids: Tensor::zeros([0], (Kind::Int64, Device::Cpu)),
    };
    let fine_map0 = Tensor::zeros([1, 128, 4, 4], (Kind::Float, Device::Cpu));
    let fine_map1 = Tensor::zeros([1, 128, 4, 4], (Kind::Float, Device::Cpu));
    let coarse_tokens0 = Tensor::zeros([1, 4, 256], (Kind::Float, Device::Cpu));
    let coarse_tokens1 = Tensor::zeros([1, 4, 256], (Kind::Float, Device::Cpu));

    let (out0, out1) = module.forward(
        &fine_map0,
        &fine_map1,
        &coarse_tokens0,
        &coarse_tokens1,
        &data,
    )?;
    assert_eq!(out0.size(), vec![0, 25, 128]);
    assert_eq!(out1.size(), vec![0, 25, 128]);
    Ok(())
}

#[test]
fn unfold_local_windows_matches_expected_patch_layout() -> Result<(), LoftrError> {
    let unfolded = unfold_local_windows(
        &Tensor::arange_start(1, 17, (Kind::Float, Device::Cpu)).view([1, 1, 4, 4]),
        3,
        2,
    )?;
    assert_eq!(unfolded.size(), vec![1, 4, 9, 1]);

    let top_left = unfolded.get(0).get(0).squeeze_dim(-1);
    let values: Vec<f32> = Vec::<f32>::try_from(top_left)?;
    for (actual, expected) in values
        .into_iter()
        .zip([0.0_f32, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 5.0, 6.0])
    {
        assert!((actual - expected).abs() < f32::EPSILON);
    }
    Ok(())
}

#[test]
fn forward_selects_only_requested_windows_without_concat() -> Result<(), LoftrError> {
    let mut config = LoftrConfig::outdoor();
    config.fine_window_size = 3;
    config.fine_concat_coarse_feat = false;
    let module = cpu_module(&config)?;
    let data = FinePreprocessData {
        hw0_f: (4, 4),
        hw0_c: (2, 2),
        b_ids: Tensor::from_slice(&[0_i64, 0]),
        i_ids: Tensor::from_slice(&[0_i64, 3]),
        j_ids: Tensor::from_slice(&[1_i64, 2]),
    };
    let fine_map0 =
        Tensor::arange_start(1, 1 + 128 * 4 * 4, (Kind::Float, Device::Cpu)).view([1, 128, 4, 4]);
    let fine_map1 = (&fine_map0 + 10_000.0).shallow_clone();
    let coarse_tokens0 = Tensor::zeros([1, 4, 256], (Kind::Float, Device::Cpu));
    let coarse_tokens1 = Tensor::zeros([1, 4, 256], (Kind::Float, Device::Cpu));

    let (out0, out1) = module.forward(
        &fine_map0,
        &fine_map1,
        &coarse_tokens0,
        &coarse_tokens1,
        &data,
    )?;
    assert_eq!(out0.size(), vec![2, 9, 128]);
    assert_eq!(out1.size(), vec![2, 9, 128]);

    let first_center = out0.get(0).get(4).double_value(&[0]);
    let second_center = out0.get(1).get(4).double_value(&[0]);
    let first_right_center = out1.get(0).get(4).double_value(&[0]);
    let second_right_center = out1.get(1).get(4).double_value(&[0]);

    assert!((first_center - 1.0).abs() < f64::EPSILON);
    assert!((second_center - 11.0).abs() < f64::EPSILON);
    assert!((first_right_center - 10_003.0).abs() < f64::EPSILON);
    assert!((second_right_center - 10_009.0).abs() < f64::EPSILON);
    Ok(())
}

#[test]
fn forward_with_concat_keeps_output_shape() -> Result<(), LoftrError> {
    let config = LoftrConfig::outdoor();
    let module = cpu_module(&config)?;
    let data = FinePreprocessData {
        hw0_f: (4, 4),
        hw0_c: (2, 2),
        b_ids: Tensor::from_slice(&[0_i64, 0, 0]),
        i_ids: Tensor::from_slice(&[0_i64, 1, 3]),
        j_ids: Tensor::from_slice(&[1_i64, 2, 0]),
    };
    let fine_map0 = Tensor::randn([1, 128, 4, 4], (Kind::Float, Device::Cpu));
    let fine_map1 = Tensor::randn([1, 128, 4, 4], (Kind::Float, Device::Cpu));
    let coarse_tokens0 = Tensor::randn([1, 4, 256], (Kind::Float, Device::Cpu));
    let coarse_tokens1 = Tensor::randn([1, 4, 256], (Kind::Float, Device::Cpu));

    let (out0, out1) = module.forward(
        &fine_map0,
        &fine_map1,
        &coarse_tokens0,
        &coarse_tokens1,
        &data,
    )?;
    assert_eq!(out0.size(), vec![3, 25, 128]);
    assert_eq!(out1.size(), vec![3, 25, 128]);
    Ok(())
}

#[test]
fn stride_matches_kornia_height_axis_behavior() -> Result<(), LoftrError> {
    let data = FinePreprocessData {
        hw0_f: (8, 12),
        hw0_c: (4, 4),
        b_ids: Tensor::zeros([0], (Kind::Int64, Device::Cpu)),
        i_ids: Tensor::zeros([0], (Kind::Int64, Device::Cpu)),
        j_ids: Tensor::zeros([0], (Kind::Int64, Device::Cpu)),
    };
    assert_eq!(data.stride()?, 2);
    Ok(())
}
