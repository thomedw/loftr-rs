use super::*;

#[test]
fn create_position_encoding_has_expected_shape() -> Result<(), LoftrError> {
    let pe = create_position_encoding(256, (32, 48), false, Device::Cpu)?;
    assert_eq!(pe.size(), vec![1, 256, 32, 48]);
    Ok(())
}

#[test]
fn forward_expands_internal_encoding_when_input_is_larger() -> Result<(), LoftrError> {
    let mut module = PositionEncodingSine::new(256, (8, 8), false, Device::Cpu)?;
    let input = Tensor::zeros([1, 256, 16, 12], (Kind::Float, Device::Cpu));
    let out = module.forward(&input)?;
    assert_eq!(out.size(), vec![1, 256, 16, 12]);
    assert!(module.pe.size()[2] >= 16);
    assert!(module.pe.size()[3] >= 12);
    Ok(())
}

#[test]
fn temp_bug_fix_changes_encoding_values() -> Result<(), LoftrError> {
    let fixed = create_position_encoding(256, (4, 4), true, Device::Cpu)?;
    let legacy = create_position_encoding(256, (4, 4), false, Device::Cpu)?;
    let diff = (&fixed - &legacy).abs().sum(Kind::Float).double_value(&[]);
    assert!(diff > 0.0);
    Ok(())
}

#[test]
fn forward_rejects_wrong_channel_count() {
    let mut module = match PositionEncodingSine::new(256, (8, 8), false, Device::Cpu) {
        Ok(module) => module,
        Err(err) => panic!("module construction failed unexpectedly: {err}"),
    };
    let input = Tensor::zeros([1, 128, 8, 8], (Kind::Float, Device::Cpu));
    match module.forward(&input) {
        Ok(_) => panic!("channel mismatch should fail"),
        Err(err) => assert!(format!("{err}").contains("d_model mismatch")),
    }
}
