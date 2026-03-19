use super::*;
use tch::{Device, Kind};

#[test]
fn resnet_fpn_8_2_matches_expected_output_shapes() -> Result<(), LoftrError> {
    let vs = nn::VarStore::new(Device::Cpu);
    let backbone = ResNetFpn8_2::new(&vs.root(), &LoftrConfig::outdoor().resnetfpn)?;
    let input = Tensor::randn([1, 1, 240, 320], (Kind::Float, Device::Cpu));
    let (coarse, fine) = backbone.forward_t(&input, false)?;
    assert_eq!(coarse.size(), vec![1, 256, 30, 40]);
    assert_eq!(fine.size(), vec![1, 128, 120, 160]);
    Ok(())
}

#[test]
fn build_backbone_supports_outdoor_config() -> Result<(), LoftrError> {
    let vs = nn::VarStore::new(Device::Cpu);
    let backbone = build_backbone(&vs.root(), &LoftrConfig::outdoor())?;
    let input = Tensor::randn([1, 1, 120, 160], (Kind::Float, Device::Cpu));
    let (coarse, fine) = backbone.forward_t(&input, false)?;
    assert_eq!(coarse.size(), vec![1, 256, 15, 20]);
    assert_eq!(fine.size(), vec![1, 128, 60, 80]);
    Ok(())
}

#[test]
fn build_backbone_rejects_unsupported_resolution() {
    let vs = nn::VarStore::new(Device::Cpu);
    let mut config = LoftrConfig::outdoor();
    config.resolution = (16, 4);
    match build_backbone(&vs.root(), &config) {
        Ok(_) => panic!("unsupported resolution should fail"),
        Err(err) => assert!(format!("{err}").contains("only (8, 2) is ported")),
    }
}
